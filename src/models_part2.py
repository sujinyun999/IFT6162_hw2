import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

# Constants from the assignment environment
ACTION_MIN = 900.0  # From CalcinerEnv (u_min)
ACTION_MAX = 1300.0 # From CalcinerEnv (u_max)

TG_MIN = 900.0
TG_MAX = 1300.0
TS_MIN = 550.0
TS_MAX = 800.0
N_Z = 20           # Number of spatial cells
N_SPECIES = 5      # Number of chemical species
TOTAL_CHANNELS = N_SPECIES + 2  # 5 species + T_s + T_g = 7 channels

class SpatialEncoder(nn.Module):
    """
    Shared encoder backbone that extracts spatial features from the 140D state.
    It reshapes the flat input into (Batch, Channels, Length) and applies 1D Convs.
    """
    def __init__(self, hidden_dim=64, norm_params=None):
        super().__init__()
        
        self.use_fixed_norm = False
        if norm_params is not None:
            self.use_fixed_norm = True
            # Register as buffers so they are saved with the model but not trained
            self.register_buffer('state_mean', torch.FloatTensor(norm_params['state_mean']))
            self.register_buffer('state_std', torch.FloatTensor(norm_params['state_std']))
        else:
            # Fallback: Batch Norm on the 1D input before reshaping
            self.bn_input = nn.BatchNorm1d(140)

        # Convolutional layers to process spatial dependencies
        # Input: (Batch, 7, 20) -> Output: (Batch, 32, 20)
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=TOTAL_CHANNELS, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flat dimension after convolution
        # 32 channels * 20 spatial points = 640 features
        self.flat_dim = 32 * N_Z
        
        # Dense layer to compress features before the Actor/Critic heads
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, state):
        """
        Args:
            state: (Batch, 140) flat vector
        Returns:
            features: (Batch, hidden_dim) latent representation
        """
        # 0. APPLY NORMALIZATION
        if self.use_fixed_norm:
            # (x - mu) / sigma
            state = (state - self.state_mean) / (self.state_std + 1e-8)
        # else:
            # x = self.bn_input(state)


        # 1. Unpack the 140D flat vector into components based on surrogate_env.py structure
        # Structure: [Species (5*20) | Temp_Solid (20) | Temp_Gas (20)]
        batch_size = state.shape[0]
        
        # Split flat vector
        conc_len = N_SPECIES * N_Z
        concentrations = state[:, :conc_len]
        temperatures = state[:, conc_len:]
        
        # Reshape to (Batch, Channels, Length)
        # Concentrations: (Batch, 5, 20)
        conc_reshaped = concentrations.view(batch_size, N_SPECIES, N_Z)
        
        # Temperatures: (Batch, 2, 20) -> Note: env packs T_s then T_g flat
        # We need to be careful with reshaping. The env creates it via np.concatenate([T_s, T_g])
        temp_reshaped = temperatures.view(batch_size, 2, N_Z)
        
        # Concatenate along channel dimension -> (Batch, 7, 20)
        x = torch.cat([conc_reshaped, temp_reshaped], dim=1)
        
        # 2. Apply Convolutions
        x = self.conv_net(x)
        
        # 3. Flatten and project
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x



    
class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, norm_params=None):
        super(TD3Actor, self).__init__()
        
        self.encoder = SpatialEncoder(hidden_dim=hidden_dim, norm_params=norm_params)
        
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, action_dim)
        

    def forward(self, state):
        
        # 1. Extract Spatial Features
        feat = self.encoder(state)
        
        # 2. Process through MLP
        a = F.relu(self.l1(feat))
        a = torch.tanh(self.l2(a)) # Output is [-1, 1]
        
        # 3. Scale to environment bounds [action_min, action_max]
        # Formula: min + (a + 1) * 0.5 * (max - min)
        # scaled_action = ACTION_MIN + (a + 1.0) * 0.5 * (ACTION_MAX - ACTION_MIN)
        
        return a

class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, norm_params=None):
        super(TD3Critic, self).__init__()
        # Q1 Architecture
        # NOTE: We use a separate encoder for Q1 to ensure independence
        self.encoder1 = SpatialEncoder(hidden_dim=hidden_dim, norm_params=norm_params)
        self.l1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 Architecture
        self.encoder2 = SpatialEncoder(hidden_dim=hidden_dim, norm_params=norm_params)
        self.l4 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):# --- Q1 Forward ---
        s1 = self.encoder1(state)
        sa1 = torch.cat([s1, action], 1)
        
        q1 = F.relu(self.l1(sa1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # --- Q2 Forward ---
        s2 = self.encoder2(state)
        sa2 = torch.cat([s2, action], 1)
        
        q2 = F.relu(self.l4(sa2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2

    def Q1(self, state, action):
        """Used for policy updates (actor loss)"""
        s1 = self.encoder1(state)
        sa1 = torch.cat([s1, action], 1)
        
        q1 = F.relu(self.l1(sa1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    

class SpatiallyAwareActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, norm_params=None):
        super().__init__()
        # Use the spatial encoder
        self.encoder = SpatialEncoder(hidden_dim=hidden_size, norm_params=norm_params)
        
        # Actor Heads (Mean and LogStd)
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        # Extract features
        x = self.encoder(state)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # Clamp std to prevent numerical instability
        std = torch.exp(log_std.clamp(min=-20, max=2))
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        # Reparameterization trick (rsample)
        z = dist.rsample()               
        y = torch.tanh(z)                 # Squash to (-1, 1)

        # Scale to environment bounds [900, 1300]
        action_env = TG_MIN + (y + 1.0) * 0.5 * (TG_MAX - TG_MIN)

        # Calculate log_prob with Tanh correction (Enforcing boundaries)
        # Formula: log_prob(y) = log_prob(z) - log(1 - tanh(z)^2)
        log_prob = dist.log_prob(z) - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action_env, z, log_prob

    def evaluate(self, state, z):
        """
        Evaluate z (pre-tanh action) for PPO updates.
        z is used instead of the raw action to ensure the log_prob math is stable.
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        y = torch.tanh(z)
        
        log_prob = dist.log_prob(z) - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # Approx entropy
        entropy = dist.entropy().sum(-1)

        return log_prob, entropy
    
    def mean_action(self, state):
        mean, std = self.forward(state)
        y = torch.tanh(mean)                 # Squash to (-1, 1)
        # Scale to environment bounds [900, 1300]
        action_env = TG_MIN + (y + 1.0) * 0.5 * (TG_MAX - TG_MIN)

        return action_env


class SpatiallyAwareCritic(nn.Module):
    def __init__(self, state_dim, hidden_size=256, norm_params=None):
        super().__init__()
        # Re-use the same encoder architecture (but different weights)
        self.encoder = SpatialEncoder(hidden_dim=hidden_size, norm_params=norm_params)
        
        # Value Head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.encoder(state)
        value = self.value_head(x)
        return value
