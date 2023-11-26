import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
    
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m, std=np.sqrt(2), bias_const=0.0):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, std)
        torch.nn.init.constant_(m.bias, bias_const)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_actions):
        super(ActorCritic, self).__init__()
        self.critic = VNetwork(num_inputs, hidden_dim)
        self.actor = GaussianPolicy(num_inputs, num_actions, hidden_dim)
    
    def forward(self, state):
        mean, log_std = self.actor(state)
        std = log_std.exp()
        distribution = Normal(mean, std)
        value = self.critic(state)
        return distribution, value

    def sample(self, state):
        distribution, value = self(state)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob.sum(dim=1), entropy.sum(dim=1), value
    
    def get_value(self, state):
        return self.critic(state)
    
    def act(self, state):
        mean, _ = self.actor(state)
        return mean

class VNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(VNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)
        weights_init_(self.linear3, 1.0)

    def forward(self, state):
        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x.squeeze(-1)
    
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Parameter(
            torch.zeros((1, num_actions), requires_grad=True), requires_grad=True)

        self.apply(weights_init_)
        weights_init_(self.mean_linear, 0.01)
            
    def forward(self, state):
        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std.expand_as(mean)

        return mean, log_std
