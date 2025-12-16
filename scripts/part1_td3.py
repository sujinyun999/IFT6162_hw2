import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import sys
import copy

# Add parent directory to path to import calciner package
sys.path.insert(0, str(Path(__file__).parent.parent))

from calciner import CalcinerEnv, evaluate_baseline, ConstantTemperatureController
from src.utils import set_seed, plot_learning_curves

# --- Hyperparameters ---
SEED = 0
HIDDEN_SIZE = 256           # TD3 often benefits from slightly larger networks
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
GAMMA = 0.99                # Discount factor
TAU = 0.005                 # Soft update parameter
POLICY_NOISE = 0.2          # Noise added to target policy during critic update (smoothing)
NOISE_CLIP = 0.5            # Range to clip target policy noise
EXPLORATION_NOISE = 0.1     # Noise added to action during collection
POLICY_DELAY = 2            # Delay freq for actor updates
BATCH_SIZE = 256
MAX_TIMESTEPS = 20000       # Total training timesteps (approx 500 episodes)
START_TIMESTEPS = 1000      # Random actions before training starts
BUFFER_SIZE = 100000        # Replay buffer capacity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_td3_checkpoint(path: Path, agent, episode_num: int, t: int, history: dict, best_avg_reward: float):
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "episode_num": episode_num,
        "t": t,
        "best_avg_reward": best_avg_reward,

        "actor": agent.actor.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),

        "actor_optim": agent.actor_optimizer.state_dict(),
        "critic_optim": agent.critic_optimizer.state_dict(),

        "total_it": agent.total_it,

        "history": history,
        "losses": {
            "actor_losses": agent.actor_losses,
            "critic_losses": agent.critic_losses,
        },
    }

    torch.save(ckpt, path)
    print(f"✓ Saved TD3 checkpoint (no buffer): {path}")


# --- 1. Models (Actor & Twin Critic) ---

class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(TD3Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Tanh outputs [-1, 1], scaled to max_action
        return self.max_action * torch.tanh(self.l3(a))

class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(TD3Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# --- 2. Replay Buffer ---

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=BUFFER_SIZE):
        self.ptr = 0
        self.size = 0
        self.max_size = max_size

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(DEVICE),
            torch.FloatTensor(self.action[ind]).to(DEVICE),
            torch.FloatTensor(self.next_state[ind]).to(DEVICE),
            torch.FloatTensor(self.reward[ind]).to(DEVICE),
            torch.FloatTensor(self.not_done[ind]).to(DEVICE)
        )

# --- 3. TD3 Agent Class ---

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        
        # Initialize Actor and Critic
        self.actor = TD3Actor(state_dim, action_dim, HIDDEN_SIZE, max_action).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = TD3Critic(state_dim, action_dim, HIDDEN_SIZE).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.total_it = 0
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        
        # Metrics history
        self.critic_losses = []
        self.actor_losses = []

    def select_action(self, state, noise=0.0):
        # Expects state shape (state_dim,)
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
            
        if noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)
            
        return action.clip(-self.max_action, self.max_action)

    def train(self):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = self.replay_buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            # Select action according to policy and add clipped noise (Target Smoothing)
            noise = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value (Twin Critics)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * GAMMA * target_Q)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_losses.append(critic_loss.item())

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % POLICY_DELAY == 0:
            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_losses.append(actor_loss.item())
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# --- 4. Main Training Loop ---

def train_td3():
    set_seed(SEED)
    print(f"Starting TD3 training on {DEVICE}...")



    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    ckpt_path = ckpt_dir / "part1_td3_latest.pt"
    best_ckpt_path = ckpt_dir / "part1_td3_best.pt"


    env = CalcinerEnv(episode_length=40, dt=0.5)
    
    # Environment Wrapper Helper
    # TD3 works best with normalized actions [-1, 1]. 
    # The Env expects [900, 1300].
    # We will map Actor's [-1, 1] -> [900, 1300] inside step logic.
    ACTION_CENTER = (1300.0 + 900.0) / 2.0
    ACTION_SCALE = (1300.0 - 900.0) / 2.0  # 200.0
    
    # State dim = 3, Action dim = 1
    # Max action for internal agent is 1.0 (since we normalize)
    agent = TD3Agent(state_dim=3, action_dim=1, max_action=1.0)
    
    state = env.reset(seed=SEED)
    
    # Track metrics
    history = {'rewards': [], 'avg_rewards': []}
    ep_reward = 0
    episode_num = 0
    best_avg_reward = -float("inf")
    for t in range(MAX_TIMESTEPS):
        
        # Select Action
        if t < START_TIMESTEPS:
            # Pure exploration for the start
            action_norm = np.random.uniform(-1.0, 1.0, size=(1,))
        else:
            # Policy action + exploration noise
            action_norm = agent.select_action(state, noise=EXPLORATION_NOISE)

        # Scale action for environment: [-1, 1] -> [900, 1300]
        action_env = ACTION_CENTER + action_norm * ACTION_SCALE
        
        # Step
        next_state, reward, done, info = env.step(action_env.item())
        
        # Store in buffer (use normalized action)
        agent.replay_buffer.add(state, action_norm, next_state, reward, done)
        
        state = next_state
        ep_reward += reward

        # Train
        if t >= START_TIMESTEPS:
            agent.train()

        if done:
            history['rewards'].append(ep_reward)
            avg_reward = np.mean(history['rewards'][-20:])
            history['avg_rewards'].append(avg_reward)
            
            episode_num += 1
            if episode_num % 20 == 0:
                print(f"Step {t+1} | Episode {episode_num} | Avg Reward: {avg_reward:.2f}")
            
            # Save best checkpoint
            if (avg_reward > best_avg_reward) and episode_num > 400:
                best_avg_reward = avg_reward
                save_td3_checkpoint(best_ckpt_path, agent, episode_num, t, history, best_avg_reward)
            # Reset env
            state = env.reset()
            ep_reward = 0

    save_td3_checkpoint(ckpt_path, agent, episode_num, t, history, best_avg_reward)
    
    
    history['actor_losses'] = agent.actor_losses
    history['critic_losses'] = agent.critic_losses
    return agent, history

if __name__ == "__main__":
    trained_agent, history = train_td3()
    
    # Plotting
    plot_learning_curves(
        history['rewards'],
        history['actor_losses'],
        history['critic_losses'],
        window=20,
        save_path="plots/part1_td3_learning_curve.png",
        title="TD3 on Part1"
    )

    # --- Evaluation ---
    print("\nEvaluating Trained TD3 Agent...")

    # Wrapper to map normalized output to environment space for evaluation
    class TD3Wrapper:
        def __init__(self, agent):
            self.agent = agent
            self.center = 1100.0
            self.scale = 200.0
        
        def get_action(self, obs):
            # Select action deterministically (no noise)
            action_norm = self.agent.select_action(obs, noise=0.0)
            # Scale to [900, 1300]
            action_env = self.center + action_norm * self.scale

            return action_env.item()

    eval_controller = TD3Wrapper(trained_agent)
    env = CalcinerEnv(episode_length=40, dt=0.5)

    # Run evaluation
    results = evaluate_baseline(env, eval_controller, n_episodes=5)
    print(f"TD3 Results:")
    print(f"  Mean Energy: {results['mean_energy']:.2f}")
    print(f"  Mean Violations: {results['mean_violations']:.1f}")
    print(f"  Mean Final α: {results['mean_final_conversion']:.1%}")
    
    # Compare with Baseline
    baseline = ConstantTemperatureController(T_g_in=1261.15)
    base_results = evaluate_baseline(env, baseline, n_episodes=5)
    print(f"\nBaseline (Constant 1261K) Results:")
    print(f"  Mean Energy: {base_results['mean_energy']:.2f}")
    print(f"  Mean Violations: {base_results['mean_violations']:.1f}")
    print(f"  Mean Final α: {base_results['mean_final_conversion']:.1%}")