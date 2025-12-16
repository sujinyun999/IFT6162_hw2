import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calciner import CalcinerSimulator, N_SPECIES
from calciner.physics import plot_3d_states  # Re-use provided plotting logic if available

def run_closed_loop_physics(policy_fn, steps=100, dt=0.1):
    """
    Runs the policy against the TRUE Physics Simulator.
    
    Args:
        policy_fn: Function that takes (state_vector) -> (action_vector)
        steps: Number of simulation steps
        dt: Time step duration (default 0.1s for the simulator)
    """
    print(f"Running closed-loop validation on Physics Simulator for {steps} steps...")
    
    # Initialize Physics Simulator (True Plant)
    N_z = 20
    simulator = CalcinerSimulator(N_z=N_z, dt=dt)
    
    # Initial Condition (Cold Start)
    # Using the same logic as SurrogateCalcinerEnv.reset() for consistency
    x = simulator.sample_random_state(mode='cold') 
    
    # Storage
    history = {
        'x': [x],
        'u': [],
        't': [0.0]
    }
    
    current_x = x.copy()
    
    for k in range(steps):
        # 1. Get Action from Policy
        # Note: Ensure your policy handles the raw numpy array, 
        # or convert to torch tensor here if needed.
        with torch.no_grad():
            # Example conversion if your policy expects a tensor
            obs_tensor = torch.FloatTensor(current_x).unsqueeze(0) 
            # action = policy(obs_tensor).numpy().flatten()
            
            # GENERIC CALL - Replace with your specific policy call
            action = policy_fn(current_x) 
            
        # Clip action to valid range (Physical constraints)
        T_g_in = np.clip(action[0], 900.0, 1350.0)
        if len(action) > 1:
            T_s_in = np.clip(action[1], 550.0, 800.0)
        else:
            T_s_in = 657.15 # Default
            
        u_applied = np.array([T_g_in, T_s_in])
        
        # 2. Step the TRUE Physics Simulator
        next_x = simulator.step(current_x, u_applied)
        
        # 3. Store
        history['x'].append(next_x)
        history['u'].append(u_applied)
        history['t'].append((k + 1) * dt)
        
        current_x = next_x
        
        if (k+1) % 10 == 0:
            print(f"Step {k+1}/{steps} complete")

    return history, simulator

def plot_spatial_profiles(history, simulator):
    """
    Generates Time vs Space Heatmaps for Temperatures and Conversion.
    """
    x_data = np.array(history['x']) # Shape: (T+1, 140)
    t_data = np.array(history['t'])
    u_data = np.array(history['u'])
    
    T_steps = len(t_data)
    N_z = simulator.N_z
    
    # Unpack states for all time steps
    # simulator.vector_to_state returns (c, T_s, T_g)
    # We need to vectorize this unpacking
    
    # Pre-allocate
    T_s_map = np.zeros((N_z, T_steps))
    T_g_map = np.zeros((N_z, T_steps))
    C_kao_map = np.zeros((N_z, T_steps))
    
    for k in range(T_steps):
        c, T_s, T_g = simulator.vector_to_state(x_data[k])
        # N_z dim is usually 0 for vectors returned by unpack, but let's be safe
        # Physics arrays are usually (N_z,) or (Species, N_z)
        
        T_s_map[:, k] = T_s
        T_g_map[:, k] = T_g
        C_kao_map[:, k] = c[0] # Kaolinite is species 0
        
    z_axis = np.linspace(0, 10, N_z) # 0 to 10m
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Solid Temperature
    im1 = axes[0].pcolormesh(t_data, z_axis, T_s_map, shading='gouraud', cmap='inferno')
    axes[0].set_ylabel('Reactor Length [m]')
    axes[0].set_title('Solid Temperature $T_s(z,t)$ [K]')
    fig.colorbar(im1, ax=axes[0], label='Temp [K]')
    
    # 2. Gas Temperature
    im2 = axes[1].pcolormesh(t_data, z_axis, T_g_map, shading='gouraud', cmap='inferno')
    axes[1].set_ylabel('Reactor Length [m]')
    axes[1].set_title('Gas Temperature $T_g(z,t)$ [K]')
    fig.colorbar(im2, ax=axes[1], label='Temp [K]')
    
    # 3. Kaolinite Concentration (Conversion)
    im3 = axes[2].pcolormesh(t_data, z_axis, C_kao_map, shading='gouraud', cmap='viridis_r')
    axes[2].set_ylabel('Reactor Length [m]')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_title('Kaolinite Concentration (reactant)')
    fig.colorbar(im3, ax=axes[2], label='Conc [mol/m3]')
    
    plt.tight_layout()
    plt.savefig('figures/closed_loop_profiles.png', dpi=150)
    print("Saved spatial profiles to figures/closed_loop_profiles.png")
    
    # --- Check for Sharp Fronts ---
    # Calculate spatial gradients at final time step
    T_s_final = T_s_map[:, -1]
    grad_Ts = np.abs(np.gradient(T_s_final, z_axis[1]-z_axis[0]))
    max_grad = np.max(grad_Ts)
    print(f"\nAnalysis:")
    print(f"Max Solid Temp Gradient: {max_grad:.2f} K/m")
    if max_grad > 200:
        print("WARNING: High thermal gradients detected! Potential for thermal shock.")
    else:
        print("Gradients appear smooth.")

# ==========================================
# Usage Example
# ==========================================

if __name__ == "__main__":
    # 1. Load your trained agent here
    # Example: wrapper for a random agent or your loaded torch model
    class RandomPolicy:
        def __call__(self, state):
            # Return T_g_in, T_s_in
            return np.array([1200.0, 650.0]) + np.random.randn(2)*10

    # Replace this with: my_policy = torch.load('best_model.pt')
    my_policy = RandomPolicy() 

    # 2. Run Validation
    # Run for e.g. 50 steps (5 seconds)
    hist, sim = run_closed_loop_physics(my_policy, steps=50, dt=0.1)
    
    # 3. Visualize
    plot_spatial_profiles(hist, sim)