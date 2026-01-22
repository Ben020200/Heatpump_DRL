"""
Analyze Compressor Cycling Behavior

This script analyzes logged episode data to understand cycling patterns
and help calibrate the cycling_weight parameter in the reward function.
"""

import pandas as pd
import numpy as np
import os
import sys


def analyze_step_data(step_file):
    """Analyze cycling from step-level data."""
    print(f"\nAnalyzing: {step_file}")
    print("=" * 80)
    
    if not os.path.exists(step_file):
        print(f"  ❌ File not found")
        return None
    
    df = pd.read_csv(step_file)
    
    if 'action' not in df.columns:
        print(f"  ❌ No action column found")
        return None
    
    episodes = df['episode'].unique()
    results = []
    
    for ep in episodes:
        ep_data = df[df['episode'] == ep]
        actions = ep_data['action'].values
        
        # Count cycles (action changes)
        cycles = np.sum(np.diff(actions) != 0)
        
        # Episode duration
        n_steps = len(ep_data)
        hours = n_steps * 0.25  # 15-min steps
        
        # Cycle metrics
        cycles_per_hour = cycles / hours if hours > 0 else 0
        
        # Action change magnitudes
        deltas = np.abs(np.diff(actions))
        avg_magnitude = deltas.mean() if len(deltas) > 0 else 0
        max_magnitude = deltas.max() if len(deltas) > 0 else 0
        
        # Time between cycles
        avg_time_between = hours / cycles if cycles > 0 else hours
        
        # Action distribution
        action_dist = [np.sum(actions == i) for i in range(4)]
        
        results.append({
            'episode': ep,
            'total_cycles': cycles,
            'cycles_per_hour': cycles_per_hour,
            'avg_magnitude': avg_magnitude,
            'max_magnitude': max_magnitude,
            'avg_time_between_hrs': avg_time_between,
            'action_0_pct': action_dist[0] / n_steps * 100,
            'action_1_pct': action_dist[1] / n_steps * 100,
            'action_2_pct': action_dist[2] / n_steps * 100,
            'action_3_pct': action_dist[3] / n_steps * 100,
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n📊 CYCLING STATISTICS ({len(episodes)} episodes)")
    print(f"  Total Cycles:")
    print(f"    Mean: {results_df['total_cycles'].mean():.1f} ± {results_df['total_cycles'].std():.1f}")
    print(f"    Min:  {results_df['total_cycles'].min():.0f}")
    print(f"    Max:  {results_df['total_cycles'].max():.0f}")
    
    print(f"\n  Cycles Per Hour:")
    print(f"    Mean: {results_df['cycles_per_hour'].mean():.2f} ± {results_df['cycles_per_hour'].std():.2f}")
    print(f"    Min:  {results_df['cycles_per_hour'].min():.2f}")
    print(f"    Max:  {results_df['cycles_per_hour'].max():.2f}")
    
    print(f"\n  Avg Time Between Cycles:")
    print(f"    Mean: {results_df['avg_time_between_hrs'].mean():.2f} hours")
    print(f"    Min:  {results_df['avg_time_between_hrs'].min():.2f} hours")
    print(f"    Max:  {results_df['avg_time_between_hrs'].max():.2f} hours")
    
    print(f"\n  Action Change Magnitude:")
    print(f"    Avg:  {results_df['avg_magnitude'].mean():.2f} (0=no change, 3=max)")
    print(f"    Max:  {results_df['max_magnitude'].mean():.2f}")
    
    print(f"\n  Action Distribution (% of time):")
    print(f"    OFF (0):    {results_df['action_0_pct'].mean():.1f}%")
    print(f"    LOW (1):    {results_df['action_1_pct'].mean():.1f}%")
    print(f"    MEDIUM (2): {results_df['action_2_pct'].mean():.1f}%")
    print(f"    HIGH (3):   {results_df['action_3_pct'].mean():.1f}%")
    
    return results_df


def estimate_cycling_penalty_impact(cycles_per_hour, cycling_weight):
    """Estimate how much cycling contributes to reward."""
    # 48-hour episode = 192 steps
    episode_hours = 48
    total_cycles = cycles_per_hour * episode_hours
    
    # Average magnitude per cycle (assume ~1.5 for mixed changes)
    avg_magnitude = 1.5
    
    # Total cycling penalty
    cycling_penalty = -cycling_weight * avg_magnitude * total_cycles
    
    # Compare to other penalties (rough estimates)
    # Comfort: ~10 per degree deviation per step
    # Energy: ~0.005 * 4000W = 20 per step at medium power
    
    return {
        'total_cycling_penalty': cycling_penalty,
        'penalty_per_hour': cycling_penalty / episode_hours,
        'penalty_per_step': cycling_penalty / (episode_hours * 4),
    }


def recommend_cycling_weight(results_df):
    """Recommend appropriate cycling weight based on current behavior."""
    print("\n" + "=" * 80)
    print("💡 CYCLING WEIGHT RECOMMENDATIONS")
    print("=" * 80)
    
    avg_cycles_per_hour = results_df['cycles_per_hour'].mean()
    
    print(f"\nCurrent behavior: {avg_cycles_per_hour:.2f} cycles/hour")
    
    # Test different weights
    weights_to_test = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("\n╔═══════════════╦═════════════════════╦═════════════════════════╗")
    print("║ Cycling Wgt   ║  Penalty/Episode    ║  Interpretation         ║")
    print("╠═══════════════╬═════════════════════╬═════════════════════════╣")
    
    for weight in weights_to_test:
        impact = estimate_cycling_penalty_impact(avg_cycles_per_hour, weight)
        penalty = impact['total_cycling_penalty']
        
        # Categorize impact
        if abs(penalty) < 10:
            category = "Negligible 🟢"
        elif abs(penalty) < 50:
            category = "Low 🟡"
        elif abs(penalty) < 200:
            category = "Moderate 🟠"
        else:
            category = "High 🔴"
        
        print(f"║ {weight:>6.1f}        ║ {penalty:>10.1f}          ║ {category:<23} ║")
    
    print("╚═══════════════╩═════════════════════╩═════════════════════════╝")
    
    print("\n📖 Context for comparison:")
    print("  - 1°C deviation for 1 step = -10 reward")
    print("  - Running at 6kW for 1 step = -30 reward")
    print("  - Typical episode total reward: -2000 to -5000")
    
    print("\n🎯 Recommended starting points:")
    print("  - Current (0.1):  Too low - cycling essentially free")
    print("  - Conservative (1.0): Cycling costs ~1% of total reward")
    print("  - Balanced (2.0-5.0): Cycling costs ~5-10% of total reward")
    print("  - Aggressive (10.0): Strong penalty, may reduce flexibility")


def main():
    print("=" * 80)
    print("🔄 COMPRESSOR CYCLING ANALYSIS")
    print("=" * 80)
    
    # Find step data files
    log_dirs = [
        'data/logs/ppo_lit_v3',
        'data/logs/sac_lit_v3',
        'data/logs/dqn_lit_v3',
        'data/logs/ppo_literature',
        'data/logs/sac_literature',
        'data/logs/dqn_literature',
    ]
    
    all_results = {}
    
    for log_dir in log_dirs:
        step_file = os.path.join(log_dir, 'steps.csv')
        if os.path.exists(step_file):
            results = analyze_step_data(step_file)
            if results is not None:
                agent_name = os.path.basename(log_dir)
                all_results[agent_name] = results
    
    if not all_results:
        print("\n❌ No step data found in common log directories")
        print("\nSearching for any steps.csv files...")
        for root, dirs, files in os.walk('data/logs'):
            for file in files:
                if file == 'steps.csv':
                    step_file = os.path.join(root, file)
                    results = analyze_step_data(step_file)
                    if results is not None:
                        agent_name = os.path.basename(root)
                        all_results[agent_name] = results
        
        if not all_results:
            print("\n❌ Still no step data found. Run training with log_steps=True first.")
            return
    
    # Compare agents
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("📊 AGENT COMPARISON")
        print("=" * 80)
        
        comparison = []
        for agent, results in all_results.items():
            comparison.append({
                'Agent': agent,
                'Cycles/Hour': f"{results['cycles_per_hour'].mean():.2f}",
                'Total Cycles': f"{results['total_cycles'].mean():.0f}",
                'Avg Magnitude': f"{results['avg_magnitude'].mean():.2f}",
            })
        
        comp_df = pd.DataFrame(comparison)
        print(comp_df.to_string(index=False))
    
    # Provide recommendation based on first agent's data
    first_agent_results = list(all_results.values())[0]
    recommend_cycling_weight(first_agent_results)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("  1. Review cycling behavior above")
    print("  2. Choose appropriate cycling_weight in config/thermal_config.yaml")
    print("  3. Re-train agents and compare cycling reduction")
    print("  4. Use new tracking metrics in episodes.csv:")
    print("     - total_cycles")
    print("     - cycles_per_hour")
    print("     - cycling_penalty_sum")
    print()


if __name__ == '__main__':
    main()
