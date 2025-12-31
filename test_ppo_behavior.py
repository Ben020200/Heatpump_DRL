"""Quick test to understand PPO's behavior"""
import numpy as np
from stable_baselines3 import PPO
from environment.thermal_env import ThermalEnv

# Load PPO model
model = PPO.load("trained_models/ppo/ppo_cost_aware/best_model.zip")

# Create environment
env = ThermalEnv(config_path="config/thermal_config.yaml", random_weather=False)

# Run one episode and track everything
obs, info = env.reset(seed=42)
done = False
step = 0

T_indoor_history = []
T_outdoor_history = []
action_history = []
power_history = []
reward_history = []
violations = 0

print("PPO Behavior Analysis")
print("="*60)

while not done:
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    T_indoor = info['T_indoor']
    T_outdoor = info['T_outdoor']
    power = info['P_electrical']
    
    T_indoor_history.append(T_indoor)
    T_outdoor_history.append(T_outdoor)
    action_history.append(action)
    power_history.append(power)
    reward_history.append(reward)
    
    # Check violation
    if not (20.0 <= T_indoor <= 22.0):
        violations += 1
    
    step += 1

print(f"Total steps: {step}")
print(f"Total violations: {violations}")
print(f"Violation rate: {violations/step*100:.1f}%")
print()
print(f"Temperature stats:")
print(f"  Min: {min(T_indoor_history):.2f}°C")
print(f"  Max: {max(T_indoor_history):.2f}°C")
print(f"  Mean: {np.mean(T_indoor_history):.2f}°C")
print()
print(f"Action distribution:")
actions = np.array(action_history)
for i in range(4):
    count = np.sum(actions == i)
    print(f"  Action {i}: {count} ({count/len(actions)*100:.1f}%)")
print()
print(f"Energy use:")
total_energy = sum(power_history) * (900/3600) / 1000  # kWh
print(f"  Total: {total_energy:.2f} kWh")
print(f"  Mean power: {np.mean(power_history):.0f} W")
print()
print(f"Total reward: {sum(reward_history):.1f}")
print()

# Show violations by timestep
print("First 20 steps:")
print("Step | T_indoor | T_outdoor | Action | Power | Violation?")
print("-"*60)
for i in range(min(20, len(T_indoor_history))):
    viol = "YES" if not (20.0 <= T_indoor_history[i] <= 22.0) else "NO"
    print(f"{i:4d} | {T_indoor_history[i]:8.2f} | {T_outdoor_history[i]:9.2f} | "
          f"{action_history[i]:6d} | {power_history[i]:5.0f} | {viol}")
