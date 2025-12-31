"""
Visualization Utilities

Functions for plotting training progress, episode analysis,
and agent comparison.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_training_progress(episodes_df: pd.DataFrame,
                           save_path: Optional[str] = None,
                           rolling_window: int = 10):
    """
    Plot training progress over episodes.
    
    Args:
        episodes_df: DataFrame with episode data
        save_path: Path to save figure
        rolling_window: Window size for rolling average
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calculate rolling averages
    episodes_df['reward_rolling'] = episodes_df['total_reward'].rolling(
        window=rolling_window, min_periods=1
    ).mean()
    episodes_df['energy_rolling'] = episodes_df['total_energy_kwh'].rolling(
        window=rolling_window, min_periods=1
    ).mean()
    episodes_df['comfort_rolling'] = episodes_df['comfort_violation_pct'].rolling(
        window=rolling_window, min_periods=1
    ).mean()
    
    # Reward plot
    ax = axes[0, 0]
    ax.plot(episodes_df['episode'], episodes_df['total_reward'], alpha=0.3, label='Episode')
    ax.plot(episodes_df['episode'], episodes_df['reward_rolling'], linewidth=2, label=f'{rolling_window}-ep avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Reward Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy consumption plot
    ax = axes[0, 1]
    ax.plot(episodes_df['episode'], episodes_df['total_energy_kwh'], alpha=0.3, label='Episode')
    ax.plot(episodes_df['episode'], episodes_df['energy_rolling'], linewidth=2, label=f'{rolling_window}-ep avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Energy Consumption')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Comfort violations plot
    ax = axes[1, 0]
    ax.plot(episodes_df['episode'], episodes_df['comfort_violation_pct'], alpha=0.3, label='Episode')
    ax.plot(episodes_df['episode'], episodes_df['comfort_rolling'], linewidth=2, label=f'{rolling_window}-ep avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Comfort Violations (%)')
    ax.set_title('Thermal Comfort Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average COP plot
    ax = axes[1, 1]
    if 'avg_cop' in episodes_df.columns:
        ax.plot(episodes_df['episode'], episodes_df['avg_cop'], alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average COP')
        ax.set_title('Heat Pump Efficiency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_episode_detail(step_data: pd.DataFrame,
                       T_comfort_min: float = 20.0,
                       T_comfort_max: float = 22.0,
                       save_path: Optional[str] = None):
    """
    Plot detailed episode analysis.
    
    Args:
        step_data: DataFrame with step-by-step data
        T_comfort_min: Minimum comfortable temperature
        T_comfort_max: Maximum comfortable temperature
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    steps = step_data['step'] if 'step' in step_data else range(len(step_data))
    hours = np.array(steps) * 0.25  # 15-minute steps to hours
    
    # Temperature plot
    ax = axes[0]
    ax.plot(hours, step_data['T_indoor'], label='Indoor', linewidth=2, color='red')
    ax.plot(hours, step_data['T_outdoor'], label='Outdoor', linewidth=2, color='blue', alpha=0.7)
    ax.axhline(T_comfort_min, color='green', linestyle='--', alpha=0.5, label='Comfort zone')
    ax.axhline(T_comfort_max, color='green', linestyle='--', alpha=0.5)
    ax.fill_between(hours, T_comfort_min, T_comfort_max, color='green', alpha=0.1)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Action plot
    ax = axes[1]
    action_colors = {0: 'gray', 1: 'yellow', 2: 'orange', 3: 'red'}
    action_labels = {0: 'OFF', 1: 'LOW', 2: 'MEDIUM', 3: 'HIGH'}
    
    for action in range(4):
        mask = step_data['action'] == action
        if mask.any():
            ax.scatter(hours[mask], step_data['action'][mask], 
                      color=action_colors[action], label=action_labels[action],
                      s=20, alpha=0.6)
    
    ax.set_ylabel('Action')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title('Heat Pump Actions')
    ax.legend(loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3)
    
    # Power and thermal output
    ax = axes[2]
    ax2 = ax.twinx()
    
    line1 = ax.plot(hours, step_data['P_electrical'] / 1000, 
                    label='Electrical Power', color='purple', linewidth=2)
    line2 = ax2.plot(hours, step_data['Q_thermal'] / 1000, 
                     label='Thermal Output', color='orange', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('Electrical Power (kW)', color='purple')
    ax2.set_ylabel('Thermal Output (kW)', color='orange')
    ax.set_title('Power and Thermal Output')
    ax.tick_params(axis='y', labelcolor='purple')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    
    # COP and reward
    ax = axes[3]
    ax2 = ax.twinx()
    
    line1 = ax.plot(hours, step_data['COP'], label='COP', color='green', linewidth=2)
    line2 = ax2.plot(hours, step_data['reward'], label='Reward', color='red', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('COP', color='green')
    ax2.set_ylabel('Reward', color='red')
    ax.set_title('Heat Pump Efficiency and Reward')
    ax.tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Episode detail plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_agent_comparison(comparison_df: pd.DataFrame,
                         metrics: List[str],
                         save_path: Optional[str] = None):
    """
    Plot comparison between different agents.
    
    Args:
        comparison_df: DataFrame with agent comparison data
        metrics: List of metrics to compare
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Get mean and std columns
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        
        if mean_col in comparison_df.columns:
            agents = comparison_df['agent']
            means = comparison_df[mean_col]
            stds = comparison_df[std_col] if std_col in comparison_df.columns else None
            
            bars = ax.bar(agents, means, yerr=stds, capsize=5, alpha=0.7)
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(agents)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            if len(agents) > 3:
                ax.set_xticklabels(agents, rotation=45, ha='right')
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Agent comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_action_distribution(step_data: pd.DataFrame,
                             save_path: Optional[str] = None):
    """
    Plot action distribution over time.
    
    Args:
        step_data: DataFrame with step data
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Action histogram
    action_counts = step_data['action'].value_counts().sort_index()
    action_labels = ['OFF', 'LOW', 'MEDIUM', 'HIGH']
    colors = ['gray', 'yellow', 'orange', 'red']
    
    ax1.bar(action_counts.index, action_counts.values, color=colors, alpha=0.7)
    ax1.set_xlabel('Action')
    ax1.set_ylabel('Count')
    ax1.set_title('Action Distribution')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(action_labels)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Action over time (heatmap-style)
    hours = np.array(range(len(step_data))) * 0.25
    
    # Create hourly bins
    hour_bins = np.arange(0, hours[-1] + 1, 1)
    action_by_hour = []
    
    for i in range(len(hour_bins) - 1):
        mask = (hours >= hour_bins[i]) & (hours < hour_bins[i + 1])
        if mask.any():
            action_by_hour.append(step_data['action'][mask].mode()[0])
        else:
            action_by_hour.append(0)
    
    # Plot as colored bars
    colors_map = {0: 'gray', 1: 'yellow', 2: 'orange', 3: 'red'}
    bar_colors = [colors_map[a] for a in action_by_hour]
    
    ax2.bar(hour_bins[:-1], np.ones(len(action_by_hour)), 
            width=1.0, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Action')
    ax2.set_title('Action Timeline')
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l, alpha=0.7) 
                      for c, l in zip(colors, action_labels)]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Action distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_training_report(log_dir: str,
                          experiment_name: str,
                          output_dir: Optional[str] = None):
    """
    Create comprehensive training report with all plots.
    
    Args:
        log_dir: Directory containing logs
        experiment_name: Name of experiment
        output_dir: Directory to save plots (defaults to log_dir/plots)
    """
    if output_dir is None:
        output_dir = os.path.join(log_dir, experiment_name, 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    episodes_path = os.path.join(log_dir, experiment_name, 'episodes.csv')
    
    if not os.path.exists(episodes_path):
        print(f"No episode data found at {episodes_path}")
        return
    
    episodes_df = pd.read_csv(episodes_path)
    
    # Training progress
    plot_training_progress(
        episodes_df,
        save_path=os.path.join(output_dir, 'training_progress.png')
    )
    
    # Load step data for last few episodes (if available)
    steps_path = os.path.join(log_dir, experiment_name, 'steps.csv')
    if os.path.exists(steps_path):
        steps_df = pd.read_csv(steps_path)
        
        # Plot last complete episode
        if len(steps_df) > 0:
            last_episode = steps_df['episode'].max()
            last_episode_data = steps_df[steps_df['episode'] == last_episode]
            
            plot_episode_detail(
                last_episode_data,
                save_path=os.path.join(output_dir, f'episode_{last_episode}_detail.png')
            )
            
            plot_action_distribution(
                last_episode_data,
                save_path=os.path.join(output_dir, f'episode_{last_episode}_actions.png')
            )
    
    print(f"✓ Training report created in {output_dir}")


if __name__ == "__main__":
    """Test visualization functions."""
    
    print("Testing Visualization Functions")
    print("=" * 60)
    
    # Create fake data for testing
    n_episodes = 50
    episodes_df = pd.DataFrame({
        'episode': range(n_episodes),
        'total_reward': -1000 + np.cumsum(np.random.randn(n_episodes) * 50),
        'total_energy_kwh': 30 + np.random.randn(n_episodes) * 5,
        'comfort_violation_pct': np.maximum(0, 20 - np.arange(n_episodes) * 0.3 + np.random.randn(n_episodes) * 2),
        'avg_cop': 3.0 + np.random.randn(n_episodes) * 0.3,
    })
    
    print("Plotting training progress...")
    plot_training_progress(episodes_df, save_path='/workspaces/Heatpump_DRL/data/test_training_progress.png')
    
    # Create fake step data
    n_steps = 192
    step_data = pd.DataFrame({
        'step': range(n_steps),
        'T_indoor': 20 + np.random.randn(n_steps) * 0.5 + 0.5 * np.sin(np.arange(n_steps) * 0.1),
        'T_outdoor': 5 + 3 * np.sin(np.arange(n_steps) * 0.05),
        'action': np.random.randint(0, 4, n_steps),
        'P_electrical': np.random.uniform(0, 6000, n_steps),
        'Q_thermal': np.random.uniform(0, 20000, n_steps),
        'COP': np.random.uniform(2.5, 4.5, n_steps),
        'reward': np.random.uniform(-50, 0, n_steps),
    })
    
    print("Plotting episode detail...")
    plot_episode_detail(step_data, save_path='/workspaces/Heatpump_DRL/data/test_episode_detail.png')
    
    print("Plotting action distribution...")
    plot_action_distribution(step_data, save_path='/workspaces/Heatpump_DRL/data/test_action_distribution.png')
    
    print("\n✓ Visualization test completed!")
