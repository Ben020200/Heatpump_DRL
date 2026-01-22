"""
Extract action distribution data from trained models for presentation.
"""

import os
import numpy as np
from stable_baselines3 import DQN, A2C, SAC
from environment.thermal_env import ThermalEnv
import yaml

def load_config():
    """Load environment configuration."""
    config_path = 'config/thermal_config.yaml'
    return config_path

def evaluate_model_actions(model, env, n_episodes=10):
    """Evaluate model and track action usage."""
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 actions: 0-3
    total_steps = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Handle both discrete (int) and continuous (array) actions
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
            
            # Ensure action is within valid range
            if action < 0 or action > 3:
                print(f"Warning: Invalid action {action}, clipping to [0, 3]")
                action = np.clip(action, 0, 3)
                
            action_counts[action] += 1
            total_steps += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    
    # Convert to percentages
    action_dist = {k: (v / total_steps * 100) for k, v in action_counts.items()}
    return action_dist, total_steps

def main():
    print("=" * 80)
    print("EXTRACTING ACTION DISTRIBUTIONS FROM TRAINED MODELS")
    print("=" * 80)
    
    config_path = load_config()
    
    # Models to evaluate - Only final 100k step runs (v3/lit_v1 versions)
    models = {
        'DQN': ('trained_models/dqn/dqn_lit_v3/dqn_model_100000_steps.zip', DQN),
        'A2C': ('trained_models/a2c/a2c_lit_v1/a2c_model_100000_steps.zip', A2C),
        'SAC': ('trained_models/sac/sac_lit_v3/sac_model_100000_steps.zip', SAC),
    }
    
    results = {}
    
    for name, (model_path, model_class) in models.items():
        if not os.path.exists(model_path):
            print(f"\n❌ {name}: Model not found at {model_path}")
            continue
        
        print(f"\n📊 Evaluating {name}...")
        
        # Load model
        model = model_class.load(model_path)
        
        # Create environment
        env = ThermalEnv(config_path)
        
        # Evaluate
        action_dist, total_steps = evaluate_model_actions(model, env, n_episodes=10)
        
        results[name] = action_dist
        
        print(f"  Total steps evaluated: {total_steps}")
        print(f"  Action distribution:")
        print(f"    OFF (0):    {action_dist[0]:5.1f}%")
        print(f"    LOW (1):    {action_dist[1]:5.1f}%")
        print(f"    MEDIUM (2): {action_dist[2]:5.1f}%")
        print(f"    HIGH (3):   {action_dist[3]:5.1f}%")
    
    # Summary comparison
    if results:
        print("\n" + "=" * 80)
        print("COMPARATIVE SUMMARY")
        print("=" * 80)
        print(f"\n{'Algorithm':<10} {'OFF (0)':<12} {'LOW (1)':<12} {'MED (2)':<12} {'HIGH (3)':<12}")
        print("-" * 60)
        for name, dist in results.items():
            print(f"{name:<10} {dist[0]:5.1f}%      {dist[1]:5.1f}%      {dist[2]:5.1f}%      {dist[3]:5.1f}%")
        
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)
        
        for name, dist in results.items():
            on_pct = dist[1] + dist[2] + dist[3]
            low_med_pct = dist[1] + dist[2]
            print(f"\n{name}:")
            print(f"  - Heat pump ON: {on_pct:.1f}% of time")
            print(f"  - LOW/MEDIUM usage: {low_med_pct:.1f}%")
            print(f"  - HIGH usage: {dist[3]:.1f}% ({'aggressive' if dist[3] > 20 else 'conservative'})")

if __name__ == "__main__":
    main()
