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
from calciner import CalcinerEnv, ConstantTemperatureController, evaluate_baseline
from src.utils import set_seed, plot_learning_curves
from src.models_part1 import Actor, Critic


# Configuration
SEED = 0
STATE_DIM = 3
ACTION_DIM = 1
LEARNING_RATE = 1e-3
GAMMA = 0.99  # Discount factor
MAX_EPISODES = 500
HIDDEN_SIZE = 128
ACTION_MIN = 900.0  # From CalcinerEnv (u_min)
ACTION_MAX = 1300.0 # From CalcinerEnv (u_max)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: Path,
                    episode: int,
                    policy_net: nn.Module,
                    value_net: nn.Module,
                    policy_optimizer: optim.Optimizer,
                    value_optimizer: optim.Optimizer,
                    history: dict,
                    best_avg_reward: float):
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "episode": episode,
        "best_avg_reward": best_avg_reward,
        "policy_state_dict": policy_net.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "policy_optim_state_dict": policy_optimizer.state_dict(),
        "value_optim_state_dict": value_optimizer.state_dict(),
        "history": history,
        # optional reproducibility
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }

    torch.save(ckpt, path)
    print(f"✓ Saved checkpoint: {path} (episode={episode}, best_avg_reward={best_avg_reward:.4f})")




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
    
    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    ckpt_path = ckpt_dir / "part1_reinforce_latest.pt"
    best_ckpt_path = ckpt_dir / "part1_reinforce_best.pt"

    # Initialize environment and networks
    env = CalcinerEnv(episode_length=40, dt=0.5)
    policy_net = Actor(STATE_DIM, ACTION_DIM, HIDDEN_SIZE).to(DEVICE)
    value_net = Critic(STATE_DIM, HIDDEN_SIZE).to(DEVICE)
    
    # Optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    value_optimizer = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()
    
    # Baseline for performance comparison (optional but good practice)
    baseline_results = evaluate_baseline(env, ConstantTemperatureController(T_g_in=1261.15), n_episodes=5)
    print("\n--- Baseline Performance (T=1261K) ---")
    print(f"Mean Energy: {baseline_results['mean_energy']:.2f}")
    print(f"Mean Violations: {baseline_results['mean_violations']:.1f}")
    print(f"Mean Final α: {baseline_results['mean_final_conversion']:.1%}")
    print("--------------------------------------\n")


    history = {'rewards': [], 'policy_losses': [], 'value_losses': []}
    best_avg_reward = -float("inf")
    for episode in range(1, MAX_EPISODES + 1):
        # 1. Collect Trajectory
        states, actions, log_probs, rewards, values = [], [], [], [], []
        
        state_np = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state_np, dtype=torch.float32).to(DEVICE)
            
            # Policy forward pass
            action_clipped, raw_action, log_prob = policy_net.sample_action(state_tensor)

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

        avg_reward = float(np.mean(history['rewards'][-20:]))

        # Logging and Monitoring
        if episode % 20 == 0:
            avg_reward = np.mean(history['rewards'][-20:])
            print(f"Episode {episode}/{MAX_EPISODES}: "
                  f"Avg Reward = {avg_reward:7.4f} | "
                  f"Actor Loss = {policy_loss.item():.4f} | "
                  f"Critic Loss = {value_loss.item():.4f}")
        
            

        # Save best checkpoint whenever avg improves
        if (avg_reward > best_avg_reward) and (episode > 400):
            best_avg_reward = avg_reward
            save_checkpoint(
                best_ckpt_path, episode,
                policy_net, value_net,
                policy_optimizer, value_optimizer,
                history, best_avg_reward
            )

    save_checkpoint(
                ckpt_path, episode,
                policy_net, value_net,
                policy_optimizer, value_optimizer,
                history, best_avg_reward
            )


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
        save_path="plots/part1_reinforce_learning_curve.png",
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
                # Use the mean of the policy for a deterministic final action
                mean, _ = self.policy(state_tensor)

                # Clip to valid range
                action = torch.clamp(mean, 900.0, 1300.0)
            return action.item()


    final_controller = TrainedPolicyController(trained_policy)
    final_results = evaluate_baseline(CalcinerEnv(episode_length=40, dt=0.5), final_controller, n_episodes=5)
    
    print(f"Mean Energy: {final_results['mean_energy']:.2f}")
    print(f"Mean Violations: {final_results['mean_violations']:.1f}")
    print(f"Mean Final α: {final_results['mean_final_conversion']:.1%}")
    print("--------------------------------------------------")
    
    # You would typically plot the rewards for the submission
    print("\nReward plot data available in the 'rewards' variable for charting.")