import torch
from torch.utils.data import TensorDataset

class GAEBuffer:
    def __init__(self, num_envs, num_steps, state_dim, action_dim, device, normalize_adv=True) -> None:
        self.batch_size = num_envs * num_steps
        self.num_steps = num_steps
        self.normalize_adv = normalize_adv
        self.device = device

        
        self.states = torch.zeros((num_steps, num_envs, state_dim), dtype=torch.float, device=self.device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float, device=self.device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float, device=self.device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float, device=self.device)
        self.log_probs = torch.zeros((num_steps, num_envs), dtype=torch.float, device=self.device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float, device=self.device)

        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float, device=self.device)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float, device=self.device)
        self.pointer = 0

    def append(self, states, actions, rewards, dones, log_probs, values):
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == dones.shape[0] == log_probs.shape[0] == values.shape[0]
        self.states[self.pointer, :, :] = states
        self.actions[self.pointer, :, :] = actions
        self.rewards[self.pointer, :] = rewards
        self.dones[self.pointer, :] = dones
        self.log_probs[self.pointer, :] = log_probs
        self.values[self.pointer, :] = values
        self.pointer += 1

    def estimate_advantages(self, last_value, gamma, gae_lambda, new_values=None):
        lastgaelam = 0
        advantages = torch.zeros_like(self.rewards, dtype=torch.float, device=self.device)
        values = new_values if new_values is not None else self.values
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextvalues = last_value
            else:
                nextvalues = values[t + 1]
            nextnonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.returns = advantages + values
        # batch normalization
        if self.normalize_adv:
            adv_mean, adv_std = advantages.mean(), advantages.std() # batch normalization
            self.advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    def reset(self):
        self.pointer = 0

    def to_dataset(self):
        return TensorDataset(
            *map(
                lambda data: data.reshape(self.batch_size, *data.shape[2:]), (
                    self.states,
                    self.actions,
                    self.rewards, 
                    self.dones,
                    self.log_probs,
                    self.values,
                    self.advantages,
                    self.returns
        )))