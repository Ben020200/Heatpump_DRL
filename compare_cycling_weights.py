"""
Quick comparison script to show impact of different cycling weights
on the same episode with the same random seed.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from environment.thermal_env import ThermalEnv


def compare_cycling_weights():
    """Run same episode with different cycling weights and compare."""
    
    weights = [0.1, 1.0, 2.0, 5.0]
    results = []
    
    print("=" * 80)
    print("🔄 COMPARING CYCLING WEIGHTS ON IDENTICAL EPISODE")
    print("=" * 80)
    
    for weight in weights:
        env = ThermalEnv()
        env.cycling_weight = weight
        
        obs, info = env.reset(seed=12345)
        
        # Simple thermostat control
        T_setpoint = 21.0
        step_data = []
        
        for step in range(192):  # 48 hours
            T_indoor = obs[0]
            
            # Bang-bang control
            if T_indoor < T_setpoint - 1.0:
                action = 3
            elif T_indoor < T_setpoint - 0.5:
                action = 2
            elif T_indoor > T_setpoint + 1.0:
                action = 0
            elif T_indoor > T_setpoint + 0.5:
                action = 1
            else:
                action = env.previous_action
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_data.append({
                'step': step,
                'action': action,
                'T_indoor': info['T_indoor'],
                'reward': reward,
            })
            
            if terminated or truncated:
                break
        
        # Gather metrics
        actions = np.array([d['action'] for d in step_data])
        cycles = np.sum(np.diff(actions) != 0)
        
        result = {
            'weight': weight,
            'total_reward': env.episode_reward,
            'total_cycles': cycles,
            'cycles_per_hour': cycles / 48.0,
            'energy_kwh': env.episode_stats.get('total_energy', 0),
            'comfort_violations': env.episode_stats.get('comfort_violations', 0),
            'cycling_penalty': env.episode_stats.get('cycling_penalty_sum', 0),
            'actions': actions,
            'temperatures': [d['T_indoor'] for d in step_data],
        }
        results.append(result)
        
        print(f"\n📊 cycling_weight = {weight}")
        print(f"  Total Cycles:        {cycles}")
        print(f"  Cycles/Hour:         {cycles/48.0:.2f}")
        print(f"  Cycling Penalty:     {result['cycling_penalty']:.1f}")
        print(f"  Total Reward:        {result['total_reward']:.1f}")
        print(f"  Energy:              {result['energy_kwh']:.2f} kWh")
        print(f"  Comfort Violations:  {result['comfort_violations']}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("📊 SUMMARY COMPARISON")
    print("=" * 80)
    print()
    print("╔════════╦════════╦═════════╦═══════════╦═══════════╦══════════╦═══════════╗")
    print("║ Weight ║ Cycles ║ Cyc/Hr  ║ Cyc Pen.  ║ Tot Reward║ Energy   ║  Comfort  ║")
    print("╠════════╬════════╬═════════╬═══════════╬═══════════╬══════════╬═══════════╣")
    
    for r in results:
        print(f"║ {r['weight']:>6.1f} ║ {r['total_cycles']:>6d} ║ {r['cycles_per_hour']:>7.2f} ║ "
              f"{r['cycling_penalty']:>9.1f} ║ {r['total_reward']:>10.1f} ║ "
              f"{r['energy_kwh']:>7.2f}  ║ {r['comfort_violations']:>9d} ║")
    
    print("╚════════╩════════╩═════════╩═══════════╩═══════════╩══════════╩═══════════╝")
    
    print("\n💡 Observations:")
    print("  - Note: Cycles count is SAME because using deterministic thermostat")
    print("  - Cycling penalty increases with weight (as expected)")
    print("  - Total reward decreases as cycling is penalized more")
    print("  - RL agents WILL learn to reduce cycles with higher weights")
    print("  - Thermostat is rule-based so doesn't adapt to reward")
    
    print("\n🎯 Recommendation:")
    print(f"  Current weight (0.1): Cycling penalty = {results[0]['cycling_penalty']:.1f}")
    print(f"  Recommended (2.0):    Cycling penalty = {results[2]['cycling_penalty']:.1f}")
    print(f"  Impact: {abs(results[2]['cycling_penalty'] - results[0]['cycling_penalty']):.1f} more penalty")
    print(f"  This is {abs(results[2]['cycling_penalty'] - results[0]['cycling_penalty']) / abs(results[2]['total_reward']) * 100:.1f}% of total reward")
    
    return results


if __name__ == '__main__':
    results = compare_cycling_weights()
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete!")
    print("=" * 80)
    print("\n🔧 To apply recommended changes:")
    print("  1. Edit config/thermal_config.yaml")
    print("  2. Change cycling_weight from 0.1 to 2.0")
    print("  3. Re-train your agents")
    print("  4. Check episodes.csv for cycling reduction")
    print()
