"""
Evaluate and compare baseline controllers with RL agents.

Runs OnOff, PID, MPC, and optionally trained RL agents on the same
test episodes and compares performance metrics.
"""

import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.thermal_env import ThermalEnv
from baselines.controllers import OnOffController, PIDController, MPCController
from utils.metrics import calculate_episode_metrics
from utils.visualization import plot_agent_comparison


def continuous_to_discrete_action(continuous_action):
    """
    Convert continuous SAC action to discrete action space.
    Maps continuous [-1, 1] to discrete {0, 1, 2, 3}.
    """
    val = continuous_action[0] if hasattr(continuous_action, '__len__') else continuous_action
    if val < -0.66:
        return 0  # OFF
    elif val < 0:
        return 1  # LOW
    elif val < 0.66:
        return 2  # MEDIUM
    else:
        return 3  # HIGH


def evaluate_controller(
    controller,
    env: ThermalEnv,
    n_episodes: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Evaluate a controller over multiple episodes.
    
    Args:
        controller: Controller instance (OnOff, PID, MPC, or RL agent)
        env: Thermal environment
        n_episodes: Number of evaluation episodes
        verbose: Print progress
        
    Returns:
        Dictionary with aggregated metrics
    """
    episode_rewards = []
    episode_energies = []
    episode_costs = []
    episode_comfort_violations = []
    episode_cops = []
    episode_lengths = []
    
    # Detailed logs for first episode
    first_episode_log = None
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        if hasattr(controller, 'reset'):
            controller.reset()
        
        episode_reward = 0.0
        episode_energy = 0.0
        episode_cost = 0.0
        comfort_violations = 0
        cop_sum = 0.0
        steps = 0
        
        # For first episode, collect detailed trajectory
        if ep == 0:
            trajectory = {
                'T_indoor': [],
                'T_outdoor': [],
                'action': [],
                'power': [],
                'cost': [],
                'cop': [],
                'reward': []
            }
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Select action
            if hasattr(controller, 'predict'):  # RL agent
                action, _ = controller.predict(obs, deterministic=True)
                
                # Check if this is SAC (outputs continuous action)
                if hasattr(controller, 'policy') and hasattr(controller.policy, 'action_space'):
                    import gymnasium as gym
                    if isinstance(controller.policy.action_space, gym.spaces.Box):
                        # SAC with continuous output - convert to discrete
                        action = continuous_to_discrete_action(action)
                    else:
                        # DQN/PPO with discrete output
                        if hasattr(action, 'item'):
                            action = action.item()
                        elif hasattr(action, '__len__'):
                            action = int(action[0])
                        else:
                            action = int(action)
                else:
                    # Fallback for DQN/PPO
                    if hasattr(action, 'item'):
                        action = action.item()
                    elif hasattr(action, '__len__'):
                        action = int(action[0])
                    else:
                        action = int(action)
            else:  # Baseline controller
                action = controller.select_action(obs, info)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Accumulate metrics
            episode_reward += reward
            episode_energy += info['P_electrical'] * (env.dt / 3600.0) / 1000.0  # kWh
            episode_cost += info['electricity_cost']
            cop_sum += info['COP']
            steps += 1
            
            # Check comfort violation
            if not (env.T_comfort_min <= info['T_indoor'] <= env.T_comfort_max):
                comfort_violations += 1
            
            # Log first episode details
            if ep == 0:
                trajectory['T_indoor'].append(info['T_indoor'])
                trajectory['T_outdoor'].append(info['T_outdoor'])
                trajectory['action'].append(action)
                trajectory['power'].append(info['P_electrical'])
                trajectory['cost'].append(info['electricity_cost'])
                trajectory['cop'].append(info['COP'])
                trajectory['reward'].append(reward)
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_costs.append(episode_cost)
        episode_comfort_violations.append(comfort_violations)
        episode_cops.append(cop_sum / steps if steps > 0 else 0.0)
        episode_lengths.append(steps)
        
        if ep == 0:
            first_episode_log = trajectory
        
        if verbose:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"Reward={episode_reward:.1f}, "
                  f"Energy={episode_energy:.2f}kWh, "
                  f"Cost=€{episode_cost:.3f}, "
                  f"Violations={comfort_violations}")
    
    # Aggregate results
    results = {
        'controller_name': controller.name if hasattr(controller, 'name') else 'RL',
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'std_energy': np.std(episode_energies),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_comfort_violations': np.mean(episode_comfort_violations),
        'std_comfort_violations': np.std(episode_comfort_violations),
        'mean_cop': np.mean(episode_cops),
        'std_cop': np.std(episode_cops),
        'mean_length': np.mean(episode_lengths),
        'first_episode_log': first_episode_log,
        'all_rewards': episode_rewards,
        'all_energies': episode_energies,
        'all_costs': episode_costs,
    }
    
    return results


def compare_all_baselines(
    config_path: str,
    n_episodes: int = 10,
    rl_models: Optional[Dict] = None,
    save_dir: str = 'results/baseline_comparison'
) -> pd.DataFrame:
    """
    Compare all baseline controllers and optionally RL agents.
    
    Args:
        config_path: Path to configuration file
        n_episodes: Number of evaluation episodes
        rl_models: Dict of {name: model} for trained RL agents
        save_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = ThermalEnv(config_path=config_path, random_weather=False)
    
    # Initialize controllers
    controllers = {
        'OnOff': OnOffController(config),
        'PID': PIDController(config),
        'MPC': MPCController(config),
    }
    
    # Add RL agents if provided
    if rl_models:
        controllers.update(rl_models)
    
    # Evaluate each controller
    results_list = []
    
    print("="*70)
    print("BASELINE CONTROLLER COMPARISON")
    print("="*70)
    
    for name, controller in controllers.items():
        print(f"\nEvaluating {name} controller...")
        print("-"*70)
        
        results = evaluate_controller(controller, env, n_episodes, verbose=True)
        results_list.append(results)
        
        print(f"\n{name} Summary:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Energy: {results['mean_energy']:.3f} ± {results['std_energy']:.3f} kWh")
        print(f"  Mean Cost: €{results['mean_cost']:.4f} ± {results['std_cost']:.4f}")
        print(f"  Mean Comfort Violations: {results['mean_comfort_violations']:.1f} ± {results['std_comfort_violations']:.1f}")
        print(f"  Mean COP: {results['mean_cop']:.2f} ± {results['std_cop']:.2f}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Controller': r['controller_name'],
            'Mean Reward': r['mean_reward'],
            'Std Reward': r['std_reward'],
            'Mean Energy (kWh)': r['mean_energy'],
            'Std Energy': r['std_energy'],
            'Mean Cost (€)': r['mean_cost'],
            'Std Cost': r['std_cost'],
            'Mean Violations': r['mean_comfort_violations'],
            'Std Violations': r['std_comfort_violations'],
            'Mean COP': r['mean_cop'],
            'Std COP': r['std_cop'],
        }
        for r in results_list
    ])
    
    # Save results
    comparison_df.to_csv(os.path.join(save_dir, 'baseline_comparison.csv'), index=False)
    print(f"\n✓ Results saved to {save_dir}/baseline_comparison.csv")
    
    # Create visualization
    plot_baseline_comparison(results_list, save_dir)
    
    # Print summary table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    return comparison_df


def plot_baseline_comparison(results_list: List[Dict], save_dir: str):
    """
    Create comparison plots for all controllers.
    
    Args:
        results_list: List of results dictionaries
        save_dir: Directory to save plots
    """
    sns.set_style("whitegrid")
    
    # Prepare data
    names = [r['controller_name'] for r in results_list]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline Controller Comparison', fontsize=16, fontweight='bold')
    
    # 1. Mean Reward
    ax = axes[0, 0]
    rewards = [r['mean_reward'] for r in results_list]
    errors = [r['std_reward'] for r in results_list]
    ax.bar(names, rewards, yerr=errors, capsize=5, alpha=0.7, color='steelblue')
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Episode Reward (higher is better)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Energy Consumption
    ax = axes[0, 1]
    energies = [r['mean_energy'] for r in results_list]
    errors = [r['std_energy'] for r in results_list]
    ax.bar(names, energies, yerr=errors, capsize=5, alpha=0.7, color='coral')
    ax.set_ylabel('Energy (kWh)', fontsize=12)
    ax.set_title('Energy Consumption (lower is better)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Electricity Cost
    ax = axes[1, 0]
    costs = [r['mean_cost'] for r in results_list]
    errors = [r['std_cost'] for r in results_list]
    ax.bar(names, costs, yerr=errors, capsize=5, alpha=0.7, color='mediumseagreen')
    ax.set_ylabel('Cost (€)', fontsize=12)
    ax.set_title('Electricity Cost (lower is better)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Comfort Violations
    ax = axes[1, 1]
    violations = [r['mean_comfort_violations'] for r in results_list]
    errors = [r['std_comfort_violations'] for r in results_list]
    ax.bar(names, violations, yerr=errors, capsize=5, alpha=0.7, color='indianred')
    ax.set_ylabel('Violations (steps)', fontsize=12)
    ax.set_title('Comfort Violations (lower is better)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'baseline_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {save_dir}/baseline_comparison.png")
    plt.close()
    
    # Plot first episode trajectory for each controller
    plot_trajectories(results_list, save_dir)


def plot_trajectories(results_list: List[Dict], save_dir: str):
    """
    Plot detailed trajectories from first episode of each controller.
    
    Args:
        results_list: List of results dictionaries
        save_dir: Directory to save plots
    """
    n_controllers = len(results_list)
    fig, axes = plt.subplots(n_controllers, 1, figsize=(14, 4*n_controllers), sharex=True)
    
    if n_controllers == 1:
        axes = [axes]
    
    for idx, results in enumerate(results_list):
        ax = axes[idx]
        log = results['first_episode_log']
        
        time_hours = np.arange(len(log['T_indoor'])) * 0.25  # 15min timesteps
        
        # Plot temperature
        ax.plot(time_hours, log['T_indoor'], label='Indoor Temp', linewidth=2, color='darkred')
        ax.plot(time_hours, log['T_outdoor'], label='Outdoor Temp', linewidth=1, alpha=0.6, color='blue')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Comfort Zone')
        ax.axhline(y=22, color='green', linestyle='--', alpha=0.5)
        
        ax.set_ylabel('Temperature (°C)', fontsize=11)
        ax.set_title(f'{results["controller_name"]} Controller - Temperature Profile', 
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel('Time (hours)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'temperature_trajectories.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Trajectory plot saved to {save_dir}/temperature_trajectories.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline controllers')
    parser.add_argument('--config', type=str, default='config/thermal_config.yaml',
                        help='Path to config file')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--rl-models', type=str, nargs='+', default=None,
                        help='Paths to trained RL models (optional)')
    parser.add_argument('--rl-names', type=str, nargs='+', default=None,
                        help='Names for RL models')
    parser.add_argument('--save-dir', type=str, default='results/baseline_comparison',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load RL models if provided
    rl_models = None
    if args.rl_models:
        from stable_baselines3 import DQN, PPO, SAC
        
        rl_models = {}
        for i, model_path in enumerate(args.rl_models):
            name = args.rl_names[i] if args.rl_names and i < len(args.rl_names) else f"RL_{i}"
            
            # Try to load with different algorithms
            try:
                model = DQN.load(model_path)
                model.name = name
                rl_models[name] = model
                print(f"Loaded DQN model: {name}")
            except:
                try:
                    model = PPO.load(model_path)
                    model.name = name
                    rl_models[name] = model
                    print(f"Loaded PPO model: {name}")
                except:
                    try:
                        model = SAC.load(model_path)
                        model.name = name
                        rl_models[name] = model
                        print(f"Loaded SAC model: {name}")
                    except Exception as e:
                        print(f"Failed to load model {model_path}: {e}")
    
    # Run comparison
    comparison_df = compare_all_baselines(
        config_path=args.config,
        n_episodes=args.episodes,
        rl_models=rl_models,
        save_dir=args.save_dir
    )
    
    print(f"\n✓ Baseline comparison complete!")


if __name__ == "__main__":
    main()
