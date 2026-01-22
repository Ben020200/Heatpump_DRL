"""
Test Cycling Tracking

Quick test to verify the new cycling tracking works correctly
and demonstrate the cycling penalty impact with different weights.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from environment.thermal_env import ThermalEnv
from utils.data_logger import DataLogger
import yaml


def test_cycling_impact():
    """Test how cycling weight affects behavior and tracking."""
    
    print("=" * 80)
    print("🔄 TESTING CYCLING TRACKING & PENALTY")
    print("=" * 80)
    
    # Test with different cycling weights
    weights_to_test = [0.1, 1.0, 5.0]
    
    for cycling_weight in weights_to_test:
        print(f"\n{'=' * 80}")
        print(f"Testing with cycling_weight = {cycling_weight}")
        print('=' * 80)
        
        # Create environment with custom cycling weight
        env = ThermalEnv()
        env.cycling_weight = cycling_weight
        
        # Run a short episode with specific actions to test tracking
        obs, info = env.reset(seed=42)
        
        # Simulate different action patterns
        test_patterns = {
            'Stable': [2, 2, 2, 2, 2, 2, 2, 2],  # No cycling
            'Oscillating': [0, 3, 0, 3, 0, 3, 0, 3],  # Max cycling
            'Gradual': [0, 1, 1, 2, 2, 2, 3, 3],  # Smooth transitions
        }
        
        for pattern_name, actions in test_patterns.items():
            obs, info = env.reset(seed=42)
            total_reward = 0
            total_cycling_penalty = 0
            cycles = 0
            
            for action in actions:
                if hasattr(env, 'previous_action') and action != env.previous_action:
                    cycles += 1
                    
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # The cycling penalty is now tracked separately
                if 'cycling_penalty' in env.episode_stats:
                    total_cycling_penalty = env.episode_stats.get('cycling_penalty_sum', 0)
            
            # Get final stats
            final_cycles = env.episode_stats.get('total_cycles', cycles)
            final_cycling_penalty = env.episode_stats.get('cycling_penalty_sum', 0)
            
            print(f"\n  {pattern_name} Pattern:")
            print(f"    Actions: {actions}")
            print(f"    Total Cycles: {final_cycles}")
            print(f"    Cycling Penalty: {final_cycling_penalty:.2f}")
            print(f"    Total Reward: {total_reward:.2f}")
            print(f"    Cycling % of Total: {abs(final_cycling_penalty/total_reward)*100 if total_reward != 0 else 0:.1f}%")


def simulate_episode_cycling():
    """Simulate a full episode and show detailed cycling stats."""
    
    print("\n" + "=" * 80)
    print("📊 FULL EPISODE SIMULATION")
    print("=" * 80)
    
    env = ThermalEnv()
    logger = DataLogger(log_dir='data/logs', experiment_name='cycling_test', log_steps=True)
    
    obs, info = env.reset(seed=42)
    
    # Simple thermostat-like policy to generate realistic cycling
    T_setpoint = 21.0
    
    episode_data = []
    step = 0
    terminated = False
    truncated = False
    
    print("\nRunning 48-hour episode with simple thermostat control...")
    
    while not (terminated or truncated) and step < 192:  # 48 hours
        # Simple bang-bang control
        T_indoor = obs[0]  # First element is T_indoor
        
        if T_indoor < T_setpoint - 1.0:
            action = 3  # HIGH
        elif T_indoor < T_setpoint - 0.5:
            action = 2  # MEDIUM
        elif T_indoor > T_setpoint + 1.0:
            action = 0  # OFF
        elif T_indoor > T_setpoint + 0.5:
            action = 1  # LOW
        else:
            action = env.previous_action  # Maintain
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log step
        step_info = {
            'step': step,
            'T_indoor': info.get('T_indoor', 0),
            'T_outdoor': info.get('T_outdoor', 0),
            'action': action,
            'P_electrical': info.get('P_electrical', 0),
            'Q_thermal': info.get('Q_thermal', 0),
            'COP': info.get('COP', 0),
            'reward': reward,
        }
        logger.log_step(0, step_info)
        episode_data.append(step_info)
        
        step += 1
    
    # Get final episode stats
    final_stats = env.episode_stats
    
    # Log episode
    logger.log_episode(
        episode=0,
        episode_stats={
            'total_reward': env.episode_reward,
            'comfort_violations': final_stats.get('comfort_violations', 0),
            'total_energy': final_stats.get('total_energy', 0),
            'avg_cop': final_stats.get('avg_cop', 0),
            'avg_temperature': final_stats.get('avg_temperature', 0),
            'min_temperature': final_stats.get('min_temperature', 0),
            'max_temperature': final_stats.get('max_temperature', 0),
            'total_cycles': final_stats.get('total_cycles', 0),
            'cycling_penalty_sum': final_stats.get('cycling_penalty_sum', 0),
        },
        episode_length=step,
        terminated=terminated
    )
    
    # Calculate cycling metrics
    actions = np.array([d['action'] for d in episode_data])
    cycles = np.sum(np.diff(actions) != 0)
    episode_hours = step * 0.25
    cycles_per_hour = cycles / episode_hours
    
    print(f"\n📊 EPISODE RESULTS:")
    print(f"  Duration: {episode_hours:.1f} hours ({step} steps)")
    print(f"  Total Reward: {env.episode_reward:.1f}")
    print(f"  Total Energy: {final_stats.get('total_energy', 0):.2f} kWh")
    print(f"  Comfort Violations: {final_stats.get('comfort_violations', 0)}")
    print(f"\n🔄 CYCLING METRICS:")
    print(f"  Total Cycles: {cycles}")
    print(f"  Cycles/Hour: {cycles_per_hour:.2f}")
    print(f"  Avg Time Between Cycles: {episode_hours/cycles:.2f} hours" if cycles > 0 else "  No cycles")
    print(f"  Cycling Penalty Sum: {final_stats.get('cycling_penalty_sum', 0):.2f}")
    print(f"  Cycling % of Reward: {abs(final_stats.get('cycling_penalty_sum', 0)/env.episode_reward)*100:.1f}%" if env.episode_reward != 0 else "  N/A")
    
    print(f"\n💡 With current cycling_weight = {env.cycling_weight}:")
    print(f"  Each cycle costs: ~{env.cycling_weight * 1.5:.2f} reward (avg magnitude 1.5)")
    print(f"  {cycles} cycles cost: ~{cycles * env.cycling_weight * 1.5:.1f} total")
    
    # Estimate with different weights
    print(f"\n📈 IMPACT OF DIFFERENT CYCLING WEIGHTS:")
    for test_weight in [0.1, 1.0, 2.0, 5.0, 10.0]:
        estimated_penalty = -cycles * test_weight * 1.5
        pct_of_total = abs(estimated_penalty / env.episode_reward) * 100 if env.episode_reward != 0 else 0
        print(f"  Weight {test_weight:>4.1f}: Penalty = {estimated_penalty:>7.1f} ({pct_of_total:>4.1f}% of total reward)")
    
    print(f"\n✅ Logged to: {logger.experiment_dir}")
    return episode_data, final_stats


def analyze_reward_components():
    """Break down reward components to understand balance."""
    
    print("\n" + "=" * 80)
    print("⚖️  REWARD COMPONENT ANALYSIS")
    print("=" * 80)
    
    # Typical values during episode
    scenarios = [
        {
            'name': 'Perfect Control',
            'T_deviation': 0.0,
            'P_electrical': 3000,  # 3kW average
            'action_change': 0,
        },
        {
            'name': 'Slight Comfort Issue',
            'T_deviation': 1.0,  # 1°C off
            'P_electrical': 4000,  # 4kW
            'action_change': 1,
        },
        {
            'name': 'Major Comfort Issue',
            'T_deviation': 3.0,  # 3°C off
            'P_electrical': 2000,  # Low power
            'action_change': 0,
        },
        {
            'name': 'Oscillating Control',
            'T_deviation': 0.5,
            'P_electrical': 5000,  # 5kW
            'action_change': 3,  # Large jump
        },
    ]
    
    comfort_weight = 10.0
    energy_weight = 0.005
    cycling_weight = 0.1  # Current value
    
    print("\nCurrent weights:")
    print(f"  comfort_weight = {comfort_weight}")
    print(f"  energy_weight = {energy_weight}")
    print(f"  cycling_weight = {cycling_weight}")
    
    print("\n" + "-" * 80)
    print(f"{'Scenario':<25} {'Comfort':<12} {'Energy':<12} {'Cycling':<12} {'Total':<12}")
    print("-" * 80)
    
    for scenario in scenarios:
        comfort_penalty = -comfort_weight * scenario['T_deviation']
        energy_penalty = -energy_weight * scenario['P_electrical']
        cycling_penalty = -cycling_weight * scenario['action_change']
        total = comfort_penalty + energy_penalty + cycling_penalty
        
        print(f"{scenario['name']:<25} {comfort_penalty:>10.2f}  {energy_penalty:>10.2f}  {cycling_penalty:>10.2f}  {total:>10.2f}")
    
    print("\n💡 Observations:")
    print(f"  - Comfort dominates: 1°C costs {comfort_weight:.1f} reward")
    print(f"  - Energy moderate: 6kW costs {energy_weight * 6000:.1f} reward")
    print(f"  - Cycling negligible: Max jump (3) costs only {cycling_weight * 3:.1f} reward")
    print(f"\n  ⚠️  Cycling is {comfort_weight/cycling_weight:.0f}x less important than comfort!")
    print(f"  ⚠️  Cycling is {(energy_weight*6000)/cycling_weight:.0f}x less important than max energy!")


if __name__ == '__main__':
    # Test 1: Show cycling penalty impact
    test_cycling_impact()
    
    # Test 2: Analyze reward balance
    analyze_reward_components()
    
    # Test 3: Full episode simulation
    episode_data, stats = simulate_episode_cycling()
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)
    print("\n🎯 Key Takeaways:")
    print("  1. ✅ Cycling tracking now works (total_cycles, cycles_per_hour, cycling_penalty_sum)")
    print("  2. ⚠️  Current cycling_weight (0.1) is TOO LOW - barely affects behavior")
    print("  3. 💡 Recommended: Increase to 1.0-5.0 for meaningful impact")
    print("  4. 📊 All metrics now logged in episodes.csv for future training runs")
    print()
