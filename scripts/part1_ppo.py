import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from pathlib import Path
import sys
import random

# Add parent directory to path to import calciner package
sys.path.insert(0, str(Path(__file__).parent.parent))

from calciner import CalcinerEnv, evaluate_baseline, ConstantTemperatureController
from src.utils import set_seed, plot_learning_curves, save_checkpoint
from src.models_part1 import Actor, Critic

# --- Hyperparameters ---
SEED = 0
HIDDEN_SIZE = 128
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
GAMMA = 0.99            # Discount factor
GAE_LAMBDA = 0.95       # GAE smoothing parameter
CLIP_EPS = 0.01         # PPO clipping parameter (epsilon)
K_EPOCHS = 5          # How many times to update on the same batch
BATCH_SIZE = 128         # Mini-batch size for updates
MAX_EPISODES = 500      # Total training episodes
UPDATE_TIMESTEP = 200  # Update policy every N timesteps (or every episode)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# --- 2. PPO Memory Buffer ---

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.next_states = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.next_states[:]

    def add(self, state, action, log_prob, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.next_states.append(next_state)


# --- 3. PPO Agent Class ---

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim, HIDDEN_SIZE).to(DEVICE)
        self.critic = Critic(state_dim, HIDDEN_SIZE).to(DEVICE)
        
        # Separate optimizers usually work better
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()
        self.actor_losses = []
        self.critic_losses = []

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(DEVICE)
            # action_clipped is for Env, action_raw is for Buffer (to calculate log_prob correctly later)
            action_clipped, action_raw, action_log_prob = self.actor.sample_action(state)
            
        return action_clipped.cpu().numpy(), action_raw.cpu().numpy(), action_log_prob.cpu().numpy()

    def update(self):
        # Convert buffer to tensors
        old_states = torch.FloatTensor(np.array(self.buffer.states)).to(DEVICE)
        old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(DEVICE)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(DEVICE).squeeze()
        old_next_states = torch.FloatTensor(np.array(self.buffer.next_states)).to(DEVICE)
        dones = torch.FloatTensor(np.array(self.buffer.is_terminals)).to(DEVICE)
        rewards = torch.FloatTensor(self.buffer.rewards).to(DEVICE)

        # --- Calculate Advantages (GAE) ---
        with torch.no_grad():
            values = self.critic(old_states).squeeze().detach()
            next_values = self.critic(old_next_states).squeeze().detach()
            # TD residual (delta_t)
            deltas = rewards + GAMMA * next_values * (1.0 - dones) - values

            advantages = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
                advantages[t] = gae

            returns = advantages + values

            # normalize advantages (optional but standard)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        
        # --- PPO Update Loop (K Epochs) ---
        dataset_size = old_states.size(0)

        
        for _ in range(K_EPOCHS):
            # Mini-batch shuffle
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                idx = indices[start:end]
                
                # Get mini-batch
                mb_states = old_states[idx]
                mb_actions = old_actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_old_next_states = old_next_states[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # Evaluate current policy on old actions
                log_probs, dist_entropy = self.actor.evaluate(mb_states, mb_actions)
                state_values = self.critic(mb_states).squeeze()

                # Ratio: pi_new / pi_old
                ratios = torch.exp(log_probs - mb_old_log_probs)
                # print(ratios)
                # print("log_probs:", log_probs.shape)
                # print("mb_old_log_probs:", mb_old_log_probs.shape)

                # breakpoint()

                # Surrogate Loss
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
                loss_actor = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()

                # Value Loss
                loss_critic = self.mse_loss(state_values, mb_returns)

                self.actor_losses.append(loss_actor.item())
                self.critic_losses.append(loss_critic.item())

                # Gradient Step Actor
                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.optimizer_actor.step()

                # Gradient Step Critic
                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer_critic.step()

        self.buffer.clear()


# --- 4. Main Training Loop ---

def train_ppo():
    set_seed(SEED)
    print(f"Starting PPO training on {DEVICE}...")

    
    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    ckpt_path = ckpt_dir / "part1_ppo_latest.pt"
    best_ckpt_path = ckpt_dir / "part1_ppo_best.pt"

    
    env = CalcinerEnv(episode_length=40, dt=0.5)
    
    # State dim = 3, Action dim = 1
    agent = PPOAgent(state_dim=3, action_dim=1)
    
    # Track metrics
    history = {'rewards': [], 'avg_rewards': []}
    timestep_counter = 0

    best_avg_reward = -float("inf")
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset(seed=episode) # Seeding for reproducibility (optional)
        ep_reward = 0
        done = False
        
        while not done:
            # Select action
            # Note: We store the 'raw' action (Gaussian sample) in buffer for proper log_prob calc
            # We pass the 'clipped' action to the environment
            action_clipped, action_raw, log_prob = agent.select_action(state)

            next_state, reward, done, info = env.step(action_clipped.item())

            # Store in buffer
            agent.buffer.add(state, action_raw, log_prob, reward, done, next_state)
            
            state = next_state
            ep_reward += reward
            timestep_counter += 1
            
            # Update PPO if buffer is full enough
            if timestep_counter % UPDATE_TIMESTEP == 0:
                agent.update()

        history['rewards'].append(ep_reward)
        avg_reward = np.mean(history['rewards'][-20:])
        history['avg_rewards'].append(avg_reward)
        
        if episode % 20 == 0:
            print(f"Episode {episode}/{MAX_EPISODES} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Last Reward: {ep_reward:.4f} | "
                  f"Actor Loss: {np.mean(agent.actor_losses[-K_EPOCHS:]):.4f} | "
                  f"Critic Loss: {np.mean(agent.critic_losses[-K_EPOCHS:]):.4f}")
        
        # Save best checkpoint whenever avg improves
        if (avg_reward > best_avg_reward) and (episode > 400):
            best_avg_reward = avg_reward
            save_checkpoint(best_ckpt_path, agent, episode, timestep_counter, history, best_avg_reward)
    
    
    save_checkpoint(ckpt_path, agent, episode, timestep_counter, history, best_avg_reward)

    # Final update to ensure buffer is empty
    if len(agent.buffer.states) > 0:
        agent.update()
    
    history['policy_losses'] = agent.actor_losses
    history['value_losses'] = agent.critic_losses
        
    return agent, history

if __name__ == "__main__":
    trained_agent, history = train_ppo()
    
    plot_learning_curves(
        history['rewards'],
        history['policy_losses'],
        history['value_losses'],
        window=20,
        save_path="plots/part1_ppo_learning_curve.png",
        title="PPO on Part1"
    )

    # --- Evaluation ---
    print("\nEvaluating Trained PPO Agent...")
    
    # Create a wrapper class to fit the evaluate_baseline signature
    class PPOWrapper:
        def __init__(self, agent):
            self.agent = agent
            self.agent.actor.eval()
        
        def get_action(self, obs):
            # For evaluation, we use the deterministic mean
            with torch.no_grad():
                state = torch.FloatTensor(obs).to(DEVICE)
                mean, _ = self.agent.actor(state)
                # Clip to valid range
                action = torch.clamp(mean, 900.0, 1300.0)
            return action.item()

    eval_controller = PPOWrapper(trained_agent)
    
    # Run evaluation
    env = CalcinerEnv(episode_length=40, dt=0.5)
    results = evaluate_baseline(env, eval_controller, n_episodes=5)
    
    print(f"PPO Results:")
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