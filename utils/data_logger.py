"""
Data Logger for RL Training and Evaluation

Comprehensive logging system for tracking training progress,
episode statistics, and detailed step-by-step data.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import csv


class DataLogger:
    """
    Logger for RL training and evaluation.
    
    Logs:
        - Episode-level metrics (rewards, energy, comfort)
        - Step-level details (temperatures, actions, COP)
        - Training progress (loss, exploration rate)
    """
    
    def __init__(self, 
                 log_dir: str = "data/logs",
                 experiment_name: Optional[str] = None,
                 log_steps: bool = False):
        """
        Initialize data logger.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of experiment (auto-generated if None)
            log_steps: Whether to log detailed step data
        """
        self.log_dir = log_dir
        self.log_steps = log_steps
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # File paths
        self.episode_log_path = os.path.join(self.experiment_dir, "episodes.csv")
        self.step_log_path = os.path.join(self.experiment_dir, "steps.csv")
        self.config_path = os.path.join(self.experiment_dir, "config.json")
        self.summary_path = os.path.join(self.experiment_dir, "summary.json")
        
        # Initialize log files
        self._init_episode_log()
        if self.log_steps:
            self._init_step_log()
        
        # In-memory storage for current episode
        self.current_episode_data = []
        self.episode_count = 0
        
        print(f"DataLogger initialized: {self.experiment_dir}")
    
    def _init_episode_log(self):
        """Initialize episode log CSV."""
        if not os.path.exists(self.episode_log_path):
            with open(self.episode_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'episode',
                    'timestamp',
                    'total_reward',
                    'episode_length',
                    'avg_temperature',
                    'min_temperature',
                    'max_temperature',
                    'comfort_violations',
                    'comfort_violation_pct',
                    'total_energy_kwh',
                    'avg_power_kw',
                    'avg_cop',
                    'terminated',
                ])
    
    def _init_step_log(self):
        """Initialize step log CSV."""
        if not os.path.exists(self.step_log_path):
            with open(self.step_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'episode',
                    'step',
                    'T_indoor',
                    'T_outdoor',
                    'action',
                    'P_electrical',
                    'Q_thermal',
                    'COP',
                    'reward',
                ])
    
    def log_step(self, episode: int, step_data: Dict[str, Any]):
        """
        Log single step data.
        
        Args:
            episode: Episode number
            step_data: Dictionary with step information
        """
        if self.log_steps:
            self.current_episode_data.append({
                'episode': episode,
                'step': step_data.get('step', 0),
                'T_indoor': step_data.get('T_indoor', 0),
                'T_outdoor': step_data.get('T_outdoor', 0),
                'action': step_data.get('action', 0),
                'P_electrical': step_data.get('P_electrical', 0),
                'Q_thermal': step_data.get('Q_thermal', 0),
                'COP': step_data.get('COP', 0),
                'reward': step_data.get('reward', 0),
            })
    
    def log_episode(self, 
                    episode: int,
                    episode_stats: Dict[str, Any],
                    episode_length: int,
                    terminated: bool = False):
        """
        Log episode-level metrics.
        
        Args:
            episode: Episode number
            episode_stats: Dictionary with episode statistics
            episode_length: Number of steps in episode
            terminated: Whether episode terminated early
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate additional metrics
        comfort_violations = episode_stats.get('comfort_violations', 0)
        comfort_violation_pct = (comfort_violations / episode_length * 100) if episode_length > 0 else 0
        
        total_energy = episode_stats.get('total_energy', 0)
        avg_power = (total_energy / (episode_length * 0.25)) if episode_length > 0 else 0  # 15-min steps
        
        # Write to CSV
        with open(self.episode_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                timestamp,
                episode_stats.get('total_reward', 0),
                episode_length,
                episode_stats.get('avg_temperature', 0),
                episode_stats.get('min_temperature', 0),
                episode_stats.get('max_temperature', 0),
                comfort_violations,
                comfort_violation_pct,
                total_energy,
                avg_power,
                episode_stats.get('avg_cop', 0),
                terminated,
            ])
        
        # Write step data if logging steps
        if self.log_steps and self.current_episode_data:
            with open(self.step_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for step_data in self.current_episode_data:
                    writer.writerow([
                        step_data['episode'],
                        step_data['step'],
                        step_data['T_indoor'],
                        step_data['T_outdoor'],
                        step_data['action'],
                        step_data['P_electrical'],
                        step_data['Q_thermal'],
                        step_data['COP'],
                        step_data['reward'],
                    ])
        
        # Clear current episode data
        self.current_episode_data = []
        self.episode_count += 1
    
    def save_config(self, config: Dict[str, Any]):
        """
        Save experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_summary(self, summary: Dict[str, Any]):
        """
        Save experiment summary.
        
        Args:
            summary: Summary dictionary with final metrics
        """
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_episodes(self) -> pd.DataFrame:
        """
        Load episode log as DataFrame.
        
        Returns:
            DataFrame with episode data
        """
        if os.path.exists(self.episode_log_path):
            return pd.read_csv(self.episode_log_path)
        else:
            return pd.DataFrame()
    
    def load_steps(self) -> pd.DataFrame:
        """
        Load step log as DataFrame.
        
        Returns:
            DataFrame with step data
        """
        if os.path.exists(self.step_log_path):
            return pd.read_csv(self.step_log_path)
        else:
            return pd.DataFrame()
    
    def get_experiment_dir(self) -> str:
        """Get experiment directory path."""
        return self.experiment_dir


class EvaluationLogger:
    """
    Specialized logger for model evaluation.
    Records detailed episode data for analysis.
    """
    
    def __init__(self, save_dir: str, model_name: str):
        """
        Initialize evaluation logger.
        
        Args:
            save_dir: Directory to save evaluation results
            model_name: Name of model being evaluated
        """
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(save_dir, exist_ok=True)
        
        self.episodes = []
        
    def log_episode(self, 
                    episode_data: pd.DataFrame,
                    episode_stats: Dict[str, Any],
                    episode_num: int):
        """
        Log complete episode with all step data.
        
        Args:
            episode_data: DataFrame with step-by-step data
            episode_stats: Episode statistics
            episode_num: Episode number
        """
        # Save step data
        step_file = os.path.join(
            self.save_dir, 
            f"{self.model_name}_episode_{episode_num:03d}_steps.csv"
        )
        episode_data.to_csv(step_file, index=False)
        
        # Collect episode summary
        self.episodes.append({
            'episode': episode_num,
            'model': self.model_name,
            **episode_stats,
        })
    
    def save_summary(self):
        """Save evaluation summary."""
        if self.episodes:
            df = pd.DataFrame(self.episodes)
            summary_file = os.path.join(self.save_dir, f"{self.model_name}_summary.csv")
            df.to_csv(summary_file, index=False)
            
            # Calculate aggregate statistics
            agg_stats = {
                'model': self.model_name,
                'n_episodes': len(self.episodes),
                'mean_reward': df['total_reward'].mean() if 'total_reward' in df else 0,
                'std_reward': df['total_reward'].std() if 'total_reward' in df else 0,
                'mean_energy': df['total_energy'].mean() if 'total_energy' in df else 0,
                'mean_comfort_violations': df['comfort_violations'].mean() if 'comfort_violations' in df else 0,
                'mean_cop': df['avg_cop'].mean() if 'avg_cop' in df else 0,
            }
            
            agg_file = os.path.join(self.save_dir, f"{self.model_name}_aggregate.json")
            with open(agg_file, 'w') as f:
                json.dump(agg_stats, f, indent=2)
            
            print(f"Evaluation summary saved for {self.model_name}")
            print(f"  Mean reward: {agg_stats['mean_reward']:.2f}")
            print(f"  Mean energy: {agg_stats['mean_energy']:.2f} kWh")
            print(f"  Mean comfort violations: {agg_stats['mean_comfort_violations']:.1f}")


if __name__ == "__main__":
    """Test data logger."""
    
    print("Testing DataLogger...")
    
    # Create logger
    logger = DataLogger(log_dir="data/logs", experiment_name="test_experiment", log_steps=True)
    
    # Log some fake episodes
    for ep in range(3):
        # Log steps
        for step in range(10):
            logger.log_step(ep, {
                'step': step,
                'T_indoor': 20 + np.random.randn(),
                'T_outdoor': 5 + np.random.randn(),
                'action': np.random.randint(0, 4),
                'P_electrical': np.random.uniform(0, 6000),
                'Q_thermal': np.random.uniform(0, 20000),
                'COP': np.random.uniform(2, 5),
                'reward': np.random.uniform(-100, 0),
            })
        
        # Log episode
        logger.log_episode(
            episode=ep,
            episode_stats={
                'total_reward': np.random.uniform(-1000, 0),
                'avg_temperature': 20 + np.random.randn(),
                'min_temperature': 19,
                'max_temperature': 23,
                'comfort_violations': np.random.randint(0, 10),
                'total_energy': np.random.uniform(10, 50),
                'avg_cop': np.random.uniform(2.5, 4),
            },
            episode_length=10,
            terminated=False
        )
    
    # Load and display
    episodes_df = logger.load_episodes()
    print(f"\nLogged {len(episodes_df)} episodes")
    print(episodes_df.head())
    
    print("\n✓ DataLogger test completed!")
