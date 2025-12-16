
import numpy as np

import matplotlib.pyplot as plt

class SimulatorCalcinerEnv:
    """
    Wraps the true Physics Simulator (CalcinerSimulator) to look like a Gym environment.
    This is SLOW but necessary for the final validation step.
    """
    def __init__(self, simulator, episode_length=50, alpha_min=0.95):
        self.sim = simulator
        self.episode_length = episode_length
        self.alpha_min = alpha_min
        self.N_z = simulator.N_z
        self.c_in_nominal = 0.15 # Nominal inlet concentration
        self.t = 0
        self.state = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Use 'cold' start to match the challenging conditions of the assignment
        self.state = self.sim.sample_random_state(mode='cold')
        self.t = 0
        return self.state.copy()

    def step(self, action):
        # Action is T_g_in (scalar) or [T_g_in, T_s_in]
        # Physics simulator expects u = [T_g_in, T_s_in]
        if np.isscalar(action) or action.size == 1:
            u = np.array([float(action), 657.15]) # Default T_s_in if not controlled
        else:
            u = action

        # Clip action to physics limits (Safety)
        u[0] = np.clip(u[0], 900, 1350)
        
        # --- Step the TRUE physics (Scipy) ---
        # The simulator returns the next state as a numpy array directly
        next_state = self.sim.step(self.state, u)
        
        # --- Calculate Reward (Same logic as SurrogateEnv) ---
        # 1. Energy Cost
        energy = (u[0] - 900.0) / (1350.0 - 900.0)
        
        # 2. Conversion & Violation
        # State layout: [Conc(5x20) | T_s(20) | T_g(20)]
        # We need the last cell of the first species (Kaolinite)
        c_kao_out = next_state[self.N_z - 1] 
        alpha = 1.0 - c_kao_out / self.c_in_nominal
        alpha = np.clip(alpha, 0, 1)
        
        violation = max(0, self.alpha_min - alpha)
        penalty = 100.0 * violation ** 2
        
        reward = -energy - penalty

        self.state = next_state
        self.t += 1
        done = self.t >= self.episode_length
        
        info = {
            'alpha': alpha,
            'energy': energy,
            'violation': violation,
            'T_g_in': u[0]
        }
        
        return self.state.copy(), reward, done, info
    

def evaluate_and_collect_profiles(env, controller, n_episodes=1):
    """
    Evaluates the agent and collects full spatial state history for visualization.
    Returns metrics and the history dict of the *last* episode.
    """
    results = {
        'total_energy': [],
        'violations': [],
        'final_conversion': []
    }
    
    # Store history of the LAST episode for plotting
    profile_history = {
        'states': [],      # Full 140D states
        'actions': [],     # Control inputs
        'alphas': [],      # Conversion over time
        'time': []
    }

    for ep in range(n_episodes):
        obs = env.reset(seed=ep+100) # Different seed than training
        done = False
        
        ep_energy = 0
        ep_violation = 0
        
        # Initialize history for this episode
        current_history = {'states': [obs], 'actions': [], 'alphas': [], 'time': [0]}
        
        step_count = 0
        while not done:
            action = controller.get_action(obs)
            obs, reward, done, info = env.step(action)
            
            ep_energy += info['energy']
            ep_violation += info['violation']
            
            current_history['states'].append(obs)
            current_history['actions'].append(info['T_g_in'])
            current_history['alphas'].append(info['alpha'])
            current_history['time'].append((step_count + 1) * 0.1) # assuming dt=0.1
            
            step_count += 1
        
        # Save results
        results['total_energy'].append(ep_energy)
        results['violations'].append(ep_violation)
        results['final_conversion'].append(info['alpha'])
        
        # Keep the history of the last episode
        if ep == n_episodes - 1:
            profile_history = current_history

    metrics = {
        'mean_energy': np.mean(results['total_energy']),
        'mean_violations': np.mean(results['violations']),
        'mean_final_conversion': np.mean(results['final_conversion'])
    }
    
    return metrics, profile_history



def visualize_spatial_profiles(history, N_z=20, L=10.0, save_path="spatial_profiles.png"):
    """
    Visualizes the evolution of Temperature profiles (Ts and Tg).
    
    State Vector Structure (140D):
    0-99:    Concentrations (5 species * 20 cells)
    100-119: T_s (Solid Temp)
    120-139: T_g (Gas Temp)
    """
    states = np.array(history['states'])
    actions = np.array(history['actions'])
    times = np.array(history['time'])
    z = np.linspace(0, L, N_z)
    
    # Indices based on docs/model.md
    idx_Ts_start = 100
    idx_Tg_start = 120
    
    # Select 5 snapshots to plot (e.g., Start, Middle, End)
    snapshot_indices = np.linspace(0, len(states)-1, 5, dtype=int)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Gas Temperature Profile (Spatial)
    ax = axes[0, 0]
    for idx in snapshot_indices:
        # Extract T_g section (20 cells)
        T_g_profile = states[idx, idx_Tg_start : idx_Tg_start + N_z]
        t_val = times[idx] if idx < len(times) else times[-1]
        ax.plot(z, T_g_profile, label=f't={t_val:.1f}s')
    
    ax.set_title("Gas Temperature Profile ($T_g$)")
    ax.set_xlabel("Reactor Length [m]")
    ax.set_ylabel("Temperature [K]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Solid Temperature Profile (Spatial)
    ax = axes[0, 1]
    for idx in snapshot_indices:
        # Extract T_s section (20 cells)
        T_s_profile = states[idx, idx_Ts_start : idx_Ts_start + N_z]
        t_val = times[idx] if idx < len(times) else times[-1]
        ax.plot(z, T_s_profile, label=f't={t_val:.1f}s')
        
    ax.set_title("Solid Temperature Profile ($T_s$)")
    ax.set_xlabel("Reactor Length [m]")
    ax.set_ylabel("Temperature [K]")
    ax.grid(True, alpha=0.3)

    # 3. Control Action vs Time
    ax = axes[1, 0]
    ax.step(times[:len(actions)], actions, where='post', color='r')
    ax.axhline(1261.15, color='k', linestyle='--', label='Baseline Temp')
    ax.set_title("Control Action ($T_{g,in}$)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Inlet Temp [K]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Conversion vs Time
    ax = axes[1, 1]
    alphas = history['alphas']
    ax.plot(times[:len(alphas)], alphas, color='g', linewidth=2)
    ax.axhline(0.95, color='r', linestyle='--', label='Target (0.95)')
    ax.set_title("Outlet Conversion ($\\alpha$)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Conversion")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Spatial profile plot saved to {save_path}")
    plt.close()