
import torch.nn as nn
from torch.distributions import Normal
import torch

ACTION_MIN = 900.0  # From CalcinerEnv (u_min)
ACTION_MAX = 1300.0 # From CalcinerEnv (u_max)

class Actor(nn.Module):
    """
    Policy Network for continuous action space (Gaussian distribution).
    Outputs mean and log-std for the action, T_g_in.
    """
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Mean of the Gaussian policy
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        # Log-std of the Gaussian policy (log is more numerically stable)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.net(state)

        # 1. Squash the unbounded mean to [-1, 1] using tanh
        raw_mean = self.mean_layer(x)
        squashed_mean = torch.tanh(raw_mean)
        
        # 2. Rescale the mean from [-1, 1] to [ACTION_MIN, ACTION_MAX]
        # This ensures the mean of the policy is always in the valid range
        scaled_mean = ACTION_MIN + (squashed_mean + 1) * 0.5 * (ACTION_MAX - ACTION_MIN)
        
        # Standard deviation calculation remains the same
        log_std = self.log_std_layer(x)
        # Clamp std low to prevent collapse, high to prevent huge variance
        std = torch.exp(log_std.clamp(min=-5, max=2))

        return scaled_mean, std

    def sample_action(self, state):
        """Samples an action, clips it to the environment bounds, and returns the log probability."""
        # mean is already scaled to the correct action range
        mean, std = self.forward(state)
        dist = Normal(mean, std)
    
        # Sample from Gaussian
        raw_action = dist.rsample() # use rsample() for reparameterization

        # Clip the action to the bounds for the environment step
        action_clipped = torch.clamp(raw_action, ACTION_MIN, ACTION_MAX)

        # Calculate log-prob of the raw sample (standard REINFORCE)
        log_prob = dist.log_prob(raw_action).sum(-1, keepdim=True)

        # The SAC log-prob correction and redundant scaling are removed
        return action_clipped, raw_action, log_prob
    

    def evaluate(self, state, action):
        """Evaluate actions for PPO update (need log_prob and entropy)"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action_log_probs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        return action_log_probs, dist_entropy

class ActorV2(nn.Module):
    """
    Policy Network for continuous action space (Gaussian distribution).
    Outputs mean and log-std for the action, T_g_in.
    """
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Mean of the Gaussian policy
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        # Log-std of the Gaussian policy (log is more numerically stable)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.net(state)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clamp std low to prevent collapse, high to prevent huge variance
        std = torch.exp(log_std.clamp(min=-5, max=2))

        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        z = dist.rsample()                # pre-tanh
        y = torch.tanh(z)                 # in (-1, 1)

        # Rescale to env range
        action_env = ACTION_MIN + (y + 1.0) * 0.5 * (ACTION_MAX - ACTION_MIN)

        # Log-prob with tanh correction
        log_prob = dist.log_prob(z) - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)

        # Store 'z' (pre-tanh) and recompute the same way in evaluate
        return action_env, z, log_prob
    

    def evaluate(self, state, z):
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        y = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)
        entropy = dist.entropy().sum(-1)  # approx; true entropy after tanh is more complex

        return log_prob, entropy



class Critic(nn.Module):
    """Critic: Estimates the expected discounted return V(s)."""
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # Output: V(s) scalar
        )

    def forward(self, state):
        return self.net(state)
