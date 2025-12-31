"""Quick test of thermal environment."""
from environment.thermal_env import ThermalEnv

print("Creating thermal environment...")
env = ThermalEnv()
print("✓ Environment created successfully!")

print("\nEnvironment properties:")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

print("\nTesting reset...")
obs, info = env.reset(seed=42)
print(f"  Observation shape: {obs.shape}")
print(f"  Initial observation: {obs}")

print("\nRunning 10 random steps...")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {i+1}: T_indoor={info['T_indoor']:.1f}°C, action={action}, reward={reward:.2f}")
    if terminated or truncated:
        print("  Episode ended!")
        break

env.close()
print("\n✓ All tests passed!")
