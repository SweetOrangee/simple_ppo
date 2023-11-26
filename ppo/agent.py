from ppo.models import ActorCritic
from ppo.buffer import GAEBuffer
import gym
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time

def to_tensor(data, device):
    return torch.FloatTensor(data).to(device)

class PPOAgent:
    def __init__(
        self,
        train_env: gym.vector.VectorEnv,
        test_env: gym.Env,
        device,
        writer,
        args
    ) -> None:
        self.envs = train_env
        self.test_env = test_env
        self.num_actions = np.prod(self.envs.single_action_space.shape)
        self.num_inputs = np.prod(self.envs.single_observation_space.shape)
        self.num_envs = self.envs.num_envs
        self.device = device

        self.total_steps = args.total_steps
        self.num_steps = args.num_steps_per_epoch
        self.batch_size = self.num_steps * self.num_envs
        self.minibatch_size = args.minibatch_size
        self.num_updates = self.total_steps // self.num_steps

        self.lr = args.learning_rate
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.normalize_adv = args.normalize_adv
        self.clip_coef = args.clip_coef
        self.value_clip = args.value_clip
        self.max_grad_norm = args.max_grad_norm
        self.entropy_loss_coef = args.entropy_loss_coef
        self.value_loss_coef = args.value_loss_coef
        self.recompute_adv = args.recompute_adv

        self.actor_critic = ActorCritic(self.num_inputs, args.hidden_dim, self.num_actions).to(self.device)
        self.storage = GAEBuffer(self.num_envs, self.num_steps, self.num_inputs, self.num_actions, self.device, self.normalize_adv)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)
        self.writer = writer

    def test(self, steps):
        num_tests = 10
        average_test_return = 0
        average_test_length = 0
        for i in range(num_tests):
            done = False
            state = self.test_env.reset()
            while True:
                state = to_tensor(state, self.device).unsqueeze(0)
                action = self.actor_critic.act(state)[0, :]
                next_state, reward, done, info = self.test_env.step(action.cpu().numpy())
                state = next_state
                if "episode" in info.keys():
                    average_test_length += info["episode"]["l"]
                    average_test_return += info["episode"]["r"]
                    break
        average_test_return = average_test_return / num_tests
        average_test_length = average_test_length / num_tests
        self.writer.add_scalar("charts/average_test_return", average_test_return, steps)
        self.writer.add_scalar("charts/average_test_length", average_test_length, steps)
        print(f'steps: {steps}, average return: {average_test_return}, average length: {average_test_length}')

    def train(self):
        next_states = self.envs.reset()
        global_step = 0
        start_time = time.time()
        for update in tqdm(range(self.num_updates)):
            # linear lr annealing
            frac = 1.0 - update / self.num_updates
            lrnow = frac * self.lr
            self.optimizer.param_groups[0]["lr"] = lrnow

            self.storage.reset()
            for step in range(self.num_steps):
                states = to_tensor(next_states, self.device)
                with torch.no_grad():
                    actions, log_probs, _, values = self.actor_critic.sample(states)
                
                next_states, rewards, dones, infos = self.envs.step(actions.cpu().numpy())

                rewards = to_tensor(rewards, self.device)
                dones = to_tensor(dones, self.device)
                self.storage.append(states, actions, rewards, dones, log_probs, values)
                
                for item in infos:
                    if "episode" in item.keys():
                        self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break
                global_step += self.num_envs
            
            with torch.no_grad():
                last_values = self.actor_critic.get_value(to_tensor(next_states, self.device))
                self.storage.estimate_advantages(last_values, self.gamma, self.gae_lambda)

            dataset = self.storage.to_dataset()
            dataloader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

            pg_losses, entropy_losses, value_losses = [], [], []
            clipfracs = []
            for epoch in range(10):
                if epoch > 0 and self.recompute_adv:
                    with torch.no_grad():
                        new_values = self.actor_critic.get_value(self.storage.states)
                        last_values = self.actor_critic.get_value(to_tensor(next_states, self.device))
                        self.storage.estimate_advantages(last_values, self.gamma, self.gae_lambda, new_values=new_values)
                    dataset = self.storage.to_dataset()
                    dataloader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)
                    
                for batch in dataloader:
                    states, actions, rewards, dones, old_log_probs, values, advantages, returns = batch
                    distribution, new_values = self.actor_critic(states)
                    new_log_probs = distribution.log_prob(actions).sum(dim=1)
                    logratio = new_log_probs - old_log_probs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                    # minibatch normalization
                    # if self.normalize_adv:
                    #     adv_mean, adv_std = advantages.mean(), advantages.std() # batch normalization
                    #     advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                    
                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if self.value_clip:
                        clipped_values = values + torch.clamp(new_values - values, -self.clip_coef, self.clip_coef)
                        value_loss_clipped = (clipped_values - returns) ** 2
                        value_loss_unclipped = (new_values - returns) ** 2
                        value_loss = 0.5 * torch.max(value_loss_clipped, value_loss_unclipped).mean()
                    else:
                        value_loss = 0.5 * ((new_values - returns) ** 2).mean()
                    
                    entropy_loss = distribution.entropy().sum(dim=1).mean()
                    loss = pg_loss - self.entropy_loss_coef * entropy_loss + self.value_loss_coef * value_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    pg_losses.append(pg_loss)
                    entropy_losses.append(entropy_loss)
                    value_losses.append(value_loss)

            
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", torch.mean(torch.stack(value_losses)).item(), global_step)
            self.writer.add_scalar("losses/policy_loss", torch.mean(torch.stack(pg_losses)).item(), global_step)
            self.writer.add_scalar("losses/entropy", torch.mean(torch.stack(entropy_losses)).item(), global_step)

            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if (update + 1) % 3 == 0:
                with torch.no_grad():
                    self.test(global_step)
                

            


        