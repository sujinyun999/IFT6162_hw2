#!/usr/bin/env python3
"""
Demo: Verify your setup works

Run this to test both the simplified environment (Part 1) 
and the neural surrogate (Part 2).
"""

import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from calciner import (
    CalcinerEnv, 
    ConstantTemperatureController,
    evaluate_baseline,
    CalcinerSimulator,
    SpatiallyAwareDynamics,
    SurrogateModel,
    SurrogateCalcinerEnv,
)


def test_part1():
    """Test the simplified RL environment."""
    print("=" * 60)
    print("Part 1: Simplified Environment")
    print("=" * 60)
    
    env = CalcinerEnv(episode_length=40, dt=0.5)
    
    print(f"\nEnvironment specs:")
    print(f"  State dim: 3 (alpha, alpha_min, t/T)")
    print(f"  Action: T_g_in ∈ [900, 1300] K")
    print(f"  Episode length: {env.episode_length} steps")
    
    # Run one episode with random actions
    obs = env.reset(seed=42)
    print(f"\nInitial state: {obs}")
    
    total_reward = 0
    for step in range(10):
        action = np.random.uniform(1000, 1200)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(total_reward)
        if step < 5:
            print(f"  Step {step}: α={info['alpha']:.2%}, u={action:.0f}K, r={reward:.2f}")
    print(f"  ...")
    
    # Evaluate baseline
    baseline = ConstantTemperatureController(T_g_in=1261.15)
    results = evaluate_baseline(env, baseline, n_episodes=3)
    
    print(f"\nConstant-temperature baseline (T=1261K):")
    print(f"  Mean energy: {results['mean_energy']:.2f}")
    print(f"  Mean violations: {results['mean_violations']:.1f}")
    print(f"  Mean final α: {results['mean_final_conversion']:.1%}")
    
    print("\n✓ Part 1 environment works!")
    return True


def test_part2():
    """Test the neural surrogate for the full problem."""
    print("\n" + "=" * 60)
    print("Part 2: Neural Surrogate (140D)")
    print("=" * 60)
    
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
    
    # Test surrogate forward pass
    x0 = simulator.sample_random_state()
    u = np.array([1200.0, 700.0])  # T_g_in, T_s_in
    
    x0_torch = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
    u_torch = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        x1_surr = surrogate.step(x0_torch, u_torch)
    
    # Compare to physics
    x1_phys = simulator.step(x0, u)
    
    # Compute conversion (kaolinite at outlet)
    c_kao_out_surr = x1_surr[0, N_z - 1].item()
    c_kao_out_phys = x1_phys[N_z - 1]
    
    print(f"\nSingle-step test (T_g_in = 1200K):")
    print(f"  Physics c_kao[outlet]: {c_kao_out_phys:.4f} mol/m³")
    print(f"  Surrogate c_kao[outlet]: {c_kao_out_surr:.4f} mol/m³")
    print(f"  Relative error: {abs(c_kao_out_surr - c_kao_out_phys) / (c_kao_out_phys + 1e-6):.1%}")
    
    # Timing comparison
    import time
    
    n_runs = 100
    start = time.time()
    for _ in range(n_runs):
        _ = simulator.step(x0, u)
    phys_time = (time.time() - start) / n_runs * 1000
    
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = surrogate.step(x0_torch, u_torch)
    surr_time = (time.time() - start) / n_runs * 1000
    
    print(f"\nTiming ({n_runs} steps):")
    print(f"  Physics: {phys_time:.2f} ms/step")
    print(f"  Surrogate: {surr_time:.2f} ms/step")
    print(f"  Speedup: {phys_time / surr_time:.0f}×")
    
    # Test the RL environment wrapper
    print(f"\nTesting RL environment wrapper...")
    env = SurrogateCalcinerEnv(surrogate, episode_length=20, alpha_min=0.95)
    
    obs = env.reset(seed=42)
    print(f"  State dim: {obs.shape[0]}")
    print(f"  Action dim: {env.action_dim}")
    
    # Run a few steps
    total_reward = 0
    for step in range(5):
        action = np.random.uniform(1000, 1200)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if step == 0:
            print(f"  Step {step}: α={info['alpha']:.2%}, T_g_in={info['T_g_in']:.0f}K, r={reward:.2f}")
    
    print("\n✓ Part 2 surrogate and environment work!")
    return True


def main():
    print("\n" + "=" * 60)
    print("  Flash Calciner RL Assignment - Setup Verification")
    print("=" * 60)
    
    p1_ok = test_part1()
    p2_ok = test_part2()
    
    print("\n" + "=" * 60)
    if p1_ok and p2_ok:
        print("  ✓ All systems go! You're ready to start the assignment.")
    else:
        print("  ✗ Some tests failed. Check the output above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

