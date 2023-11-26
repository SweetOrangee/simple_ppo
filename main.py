import torch
import gym
import random
import numpy as np
import datetime
import argparse
import os
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter

from ppo.agent import PPOAgent
from envs.wrapper import NormalizeObservation, NormalizeReward
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='Ant-v3', help='the name of the gym environment')
    parser.add_argument('--experiment-name', type=str, default=None, help='the name of the experiment')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='the learning rate of the optimizer')
    parser.add_argument('--device', type=int, default=0, help='computing device')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='enable cuda acceleration')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='save the video of agent during training')
    parser.add_argument('--normalize-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='save the video of agent during training')
    parser.add_argument('--num-envs', type=int, default=1, help='the number of parallel game environments')
    parser.add_argument('--hidden-dim', type=int, default=64, help='the hidden layer dimention of the networks')
    parser.add_argument('--num-steps-per-epoch', type=int, default=2048, help='the steps of rollout for every epoch')
    parser.add_argument('--total-steps', type=int, default=5000000, help='total training timesteps')
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--gamma', type=int, default=0.99, help='the discount factor')
    parser.add_argument('--gae-lambda', type=int, default=0.95, help='the factor for the general advantage estimation')
    parser.add_argument('--clip-coef', type=int, default=0.2, help='the surrogate clipping coefficient')
    parser.add_argument('--entropy-loss-coef', type=int, default=0, help='the entropy loss coefficient')
    parser.add_argument('--value-loss-coef', type=int, default=0.5, help='the value loss coefficient')
    parser.add_argument('--value-clip', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='value function loss clipping ')
    parser.add_argument('--max-grad-norm', type=int, default=0.5, help='the norm thereshold for gradient clipping')
    parser.add_argument('--recompute-adv', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='recompute advantages before each update epochs')
    
    args = parser.parse_args()
    if args.experiment_name is not None:
        args.experiment_name = args.env_name + '_' + args.experiment_name + '_' + str(args.seed)
    else:
        args.experiment_name = args.env_name + '_' + str(args.seed)
    return args

def make_gym_env(env_name, seed, capture_video, experiment_name, train=True):
    def thunk():
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, os.path.join(f'videos/{env_name}_{experiment_name}', 'train' if train else 'test'))
        env = gym.wrappers.ClipAction(env)
        env.seed(seed)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_gym_venv(env_name, num_envs, seed=0, capture_video=False, experiment_name=None):
    if capture_video and experiment_name is None:
        timestamp = datetime.datetime.now().strftime('%y%m%d%H%M')
        experiment_name = f'{timestamp}'
    train_env = gym.vector.SyncVectorEnv(
        [make_gym_env(env_name, seed + i, i == 0 and capture_video, experiment_name) for i in range(num_envs)]
    )
    test_env = make_gym_env(env_name, seed + num_envs, capture_video, experiment_name)()
    train_env = NormalizeObservation(train_env)
    test_env = NormalizeObservation(test_env, update_rms=False)
    test_env.set_rms(train_env.get_rms())

    train_env = gym.wrappers.TransformObservation(train_env, lambda obs: np.clip(obs, -10, 10))
    test_env = gym.wrappers.TransformObservation(test_env, lambda obs: np.clip(obs, -10, 10))

    train_env = NormalizeReward(train_env)
    test_env = NormalizeReward(test_env)
    test_env.set_rms(train_env.get_rms())

    train_env = gym.wrappers.TransformReward(train_env, lambda reward: np.clip(reward, -10, 10))
    test_env = gym.wrappers.TransformReward(test_env, lambda reward: np.clip(reward, -10, 10))
    return train_env, test_env

if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_env, test_env = make_gym_venv(
        env_name=args.env_name, 
        num_envs=args.num_envs, 
        seed=args.seed,
        capture_video=args.capture_video,
        experiment_name=args.experiment_name
    )

    device = args.device if args.cuda else 'cpu'

    writer = SummaryWriter(f"runs/{args.env_name}/{args.experiment_name}")
    agent = PPOAgent(train_env, test_env, device, writer, args)

    agent.train()
    train_env.close()
    test_env.close()