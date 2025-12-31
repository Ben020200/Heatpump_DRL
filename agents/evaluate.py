"""
Model Evaluation Script

Evaluate trained RL agents and compare their performance.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN, PPO, SAC
from environment.thermal_env import ThermalEnv, DetailedLoggingWrapper
from utils.metrics import calculate_episode_metrics, compare_agents
from utils.visualization import plot_episode_detail, plot_agent_comparison
from agents.train_sac import DiscreteToBoxWrapper


def evaluate_model(model, env, n_episodes=10, verbose=True):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        env: Environment
        n_episodes: Number of evaluation episodes
        verbose: Print progress
        
    Returns:
        List of episode DataFrames, List of episode statistics
    """
    episodes_data = []
    episodes_stats = []
    
    for ep in range(n_episodes):
        if verbose:
            print(f"  Episode {ep+1}/{n_episodes}...", end='')
        
        obs, info = env.reset()
        done = False
        step_logs = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            step_logs.append({
                'step': info['episode_step'],
                'T_indoor': info['T_indoor'],
                'T_outdoor': info['T_outdoor'],
                'action': info['action'],
                'P_electrical': info['P_electrical'],
                'Q_thermal': info['Q_thermal'],
                'COP': info['COP'],
                'reward': info['reward'],
            })
        
        # Convert to DataFrame
        episode_df = pd.DataFrame(step_logs)
        episodes_data.append(episode_df)
        
        # Calculate metrics
        metrics = calculate_episode_metrics(episode_df)
        episodes_stats.append(metrics)
        
        if verbose:
            print(f" Reward: {metrics['total_reward']:.1f}, Energy: {metrics['total_energy_kwh']:.2f} kWh, "
                  f"Violations: {metrics['comfort_violations']}")
    
    return episodes_data, episodes_stats


def compare_models(model_paths, model_names, env, n_episodes=10):
    """
    Compare multiple models.
    
    Args:
        model_paths: List of paths to model files
        model_names: List of model names
        env: Environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        DataFrame with comparison results
    """
    all_results = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\nEvaluating {model_name}...")
        print("-" * 60)
        
        # Load model based on type
        if 'dqn' in model_name.lower():
            model = DQN.load(model_path)
            eval_env = env
        elif 'ppo' in model_name.lower():
            model = PPO.load(model_path)
            eval_env = env
        elif 'sac' in model_name.lower():
            model = SAC.load(model_path)
            # Wrap for SAC
            eval_env = DiscreteToBoxWrapper(env)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Evaluate
        episodes_data, episodes_stats = evaluate_model(model, eval_env, n_episodes)
        
        # Store results
        all_results[model_name] = pd.DataFrame(episodes_stats)
        
        # Print summary
        print(f"\nSummary for {model_name}:")
        print(f"  Mean reward: {all_results[model_name]['total_reward'].mean():.2f}")
        print(f"  Mean energy: {all_results[model_name]['total_energy_kwh'].mean():.2f} kWh")
        print(f"  Mean comfort violations: {all_results[model_name]['comfort_violations'].mean():.1f}")
        print(f"  Mean COP: {all_results[model_name]['avg_cop'].mean():.2f}")
    
    # Compare agents
    comparison = compare_agents(all_results)
    
    return comparison, all_results


def main():
    """Command-line interface for model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained RL agents')
    parser.add_argument('--models', nargs='+', required=True, help='Paths to model files')
    parser.add_argument('--names', nargs='+', required=True, help='Model names')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='data/evaluation', help='Output directory')
    parser.add_argument('--seed', type=int, default=999, help='Random seed')
    
    args = parser.parse_args()
    
    if len(args.models) != len(args.names):
        raise ValueError("Number of models and names must match")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    env = ThermalEnv(random_weather=True)
    env.reset(seed=args.seed)
    
    # Compare models
    comparison, all_results = compare_models(
        args.models,
        args.names,
        env,
        n_episodes=args.episodes
    )
    
    # Save comparison
    comparison_file = os.path.join(args.output, 'model_comparison.csv')
    comparison.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to: {comparison_file}")
    
    # Plot comparison
    plot_file = os.path.join(args.output, 'model_comparison.png')
    plot_agent_comparison(
        comparison,
        metrics=['total_reward', 'total_energy_kwh', 'comfort_violations', 'avg_cop'],
        save_path=plot_file
    )
    
    print(f"\n✓ Evaluation complete!")
    print(f"Results saved to: {args.output}")
    
    env.close()


if __name__ == "__main__":
    main()
