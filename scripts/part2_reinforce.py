import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal
from pathlib import Path
import sys
import math
import random



# Add the parent directory to the system path to allow imports from 'calciner'
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the environment and controller from the provided files
from calciner import (
    CalcinerEnv, 
    ConstantTemperatureController,
    evaluate_baseline,
    evaluate_baseline_part2,
    CalcinerSimulator,
    SpatiallyAwareDynamics,
    SurrogateModel,
    SurrogateCalcinerEnv,
)

from src.utils import set_seed, plot_learning_curves
from src.models_part2 import SpatiallyAwareActor, SpatiallyAwareCritic


# Configuration
SEED = 0
STATE_DIM = 140
ACTION_DIM = 1
LEARNING_RATE = 1e-3
GAMMA = 0.99  # Discount factor
MAX_EPISODES = 5000
HIDDEN_SIZE = 512
ACTION_MIN = 900.0  # From CalcinerEnv (u_min)
ACTION_MAX = 1300.0 # From CalcinerEnv (u_max)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# --- 2. Utility Functions ---

def compute_returns_and_advantage(rewards, values, gamma):
    """Compute Monte Carlo returns (G_t) and Advantage (A_t = G_t - V(s))."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    G_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE).squeeze()

    
    # Advantage = Returns - Value Estimate (Baseline)
    A_t = G_t - values.detach()

    A_t = (A_t - A_t.mean()) / (A_t.std() + 1e-8)
    
    return G_t.unsqueeze(1), A_t.unsqueeze(1)

# --- 3. REINFORCE Algorithm ---

def reinforce():
    set_seed(SEED)
    """Main training loop for REINFORCE with Baseline."""
    print(f"Starting REINFORCE training on {DEVICE}")


    model_path = Path(__file__).parent.parent / "models" / "surrogate_model.pt"
    
    if not model_path.exists():
        print(f"\n✗ Model not found at {model_path}")
        return False
    
    # Load surrogate
    checkpoint = torch.load(model_path, weights_only=False)
    N_z = checkpoint['N_z']
    dt = checkpoint['dt']
    
    model = SpatiallyAwareDynamics(N_z=N_z)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = "cuda"
    norm_params = {k: np.array(v) for k, v in checkpoint['norm_params'].items()}
    surrogate = SurrogateModel(model, norm_params, device=device)
    
    
    print(f"\nSurrogate specs:")
    print(f"  State dim: {surrogate.state_dim} ({N_z} cells × 7 vars)")
    print(f"  Control dim: {surrogate.control_dim}")
    print(f"  Time step: {dt}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Also create physics simulator for validation
    simulator = CalcinerSimulator(N_z=N_z, dt=dt)
    
    env = SurrogateCalcinerEnv(surrogate, episode_length=20, alpha_min=0.95)
    
    
    # Initialize environment and networks
    # env = CalcinerEnv(episode_length=40, dt=0.5)
    policy_net = SpatiallyAwareActor(STATE_DIM, ACTION_DIM, HIDDEN_SIZE).to(DEVICE)
    value_net = SpatiallyAwareCritic(STATE_DIM, HIDDEN_SIZE).to(DEVICE)
    
    # Optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    value_optimizer = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()
    
    # Baseline for performance comparison (optional but good practice)
    baseline_results = evaluate_baseline_part2(env, ConstantTemperatureController(T_g_in=1261.15), n_episodes=5)
    print("\n--- Baseline Performance (T=1261K) ---")
    print(f"Mean Energy: {baseline_results['mean_energy']:.2f}")
    print(f"Mean Violations: {baseline_results['mean_violations']:.1f}")
    print(f"Mean Final α: {baseline_results['mean_final_conversion']:.1%}")
    print("--------------------------------------\n")


    history = {'rewards': [], 'policy_losses': [], 'value_losses': []}
    
    for episode in range(1, MAX_EPISODES + 1):
        # 1. Collect Trajectory
        states, actions, log_probs, rewards, values = [], [], [], [], []
        
        state_np = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state_np, dtype=torch.float32).to(DEVICE)
            
            # Policy forward pass
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            action_clipped, raw_action, log_prob = policy_net.sample_action(state_tensor)
            print(action_clipped)
            # Get Value prediction (Baseline)
            value = value_net(state_tensor)
            values.append(value)

            # Env step with the action (detach for NumPy conversion)
            action_np = action_clipped.detach().cpu().numpy().item()
            # print(action_np)
            next_state_np, reward, done, info = env.step(action_np)
            
            # Store transition
            states.append(state_tensor)
            log_probs.append(log_prob.view(1))
            rewards.append(reward)
            
            state_np = next_state_np
            total_reward += reward

        history['rewards'].append(total_reward)
        # print(f"Episode {episode} Total Reward: {total_reward:.2f}")
        all_values = torch.cat(values)

        # 2. Compute Returns (G_t) and Advantage (A_t)
        G_t, A_t = compute_returns_and_advantage(rewards, all_values, GAMMA)
        log_probs_tensor = torch.cat(log_probs)
        
        # 3. Update Policy Network
        policy_optimizer.zero_grad()
        # REINFORCE policy gradient loss: - log(pi(a|s)) * A_t
        policy_loss = -(log_probs_tensor * A_t.detach()).mean()

        history['policy_losses'].append(policy_loss.item())
        policy_loss.backward()
        policy_optimizer.step()


        # 4. Update Value Network (Critic)
        value_optimizer.zero_grad()
        # Value loss: MSE between V(s) prediction and Monte Carlo return G_t

        value_loss = mse_loss(all_values, G_t)

        history['value_losses'].append(value_loss.item())
        value_loss.backward()
        value_optimizer.step()
        
        # Logging and Monitoring
        if episode % 20 == 0:
            avg_reward = np.mean(history['rewards'][-20:])
            print(f"Episode {episode}/{MAX_EPISODES}: "
                  f"Avg Reward = {avg_reward:7.4f} | "
                  f"Actor Loss = {policy_loss.item():.4f} | "
                  f"Critic Loss = {value_loss.item():.4f}")

    print("\nREINFORCE Training Finished.")
    # Return the trained policy network
    return policy_net, history

if __name__ == "__main__":
    trained_policy, history = reinforce()
    
    plot_learning_curves(
        history['rewards'],
        history['policy_losses'],
        history['value_losses'],
        window=20,
        save_path="plots/reinforce_learning_curve.png",
        title="REINFORCE with baseline on Part1"
    )

    # --- Final Evaluation (Optional) ---
    print("\n--- Final Policy Evaluation (5 Episodes) ---")
    
    class TrainedPolicyController:
        def __init__(self, policy):
            self.policy = policy
            self.policy.eval()

        def get_action(self, obs: np.ndarray) -> float:
            with torch.no_grad():
                state_tensor = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
                state_tensor = state_tensor.unsqueeze(0)
                # Use the mean of the policy for a deterministic final action
                mean, _ = self.policy(state_tensor)

                # Clip to valid range
                action = torch.clamp(mean, 900.0, 1300.0)
                print("Final action:", action)
            return action.item()


    model_path = Path(__file__).parent.parent / "models" / "surrogate_model.pt"
    
    if not model_path.exists():
        print(f"\n✗ Model not found at {model_path}")
        raise FileNotFoundError
    
    # Load surrogate
    checkpoint = torch.load(model_path, weights_only=False)
    N_z = checkpoint['N_z']
    dt = checkpoint['dt']
    
    model = SpatiallyAwareDynamics(N_z=N_z)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = "cuda"
    norm_params = {k: np.array(v) for k, v in checkpoint['norm_params'].items()}
    surrogate = SurrogateModel(model, norm_params, device=device)
    
    
    print(f"\nSurrogate specs:")
    print(f"  State dim: {surrogate.state_dim} ({N_z} cells × 7 vars)")
    print(f"  Control dim: {surrogate.control_dim}")
    print(f"  Time step: {dt}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Also create physics simulator for validation
    simulator = CalcinerSimulator(N_z=N_z, dt=dt)
    env = SurrogateCalcinerEnv(surrogate, episode_length=20, alpha_min=0.95)
    

    final_controller = TrainedPolicyController(trained_policy)
    final_results = evaluate_baseline_part2(env, final_controller, n_episodes=5)

    
    print(f"Mean Energy: {final_results['mean_energy']:.2f}")
    print(f"Mean Violations: {final_results['mean_violations']:.1f}")
    print(f"Mean Final α: {final_results['mean_final_conversion']:.1%}")
    print("--------------------------------------------------")
    
    # You would typically plot the rewards for the submission
    print("\nReward plot data available in the 'rewards' variable for charting.")