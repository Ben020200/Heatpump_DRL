"""
A2C Training Script

Train an Advantage Actor-Critic (A2C) agent for heat pump control.
A2C is a simpler on-policy algorithm that often works well as a baseline.
"""

import os
import sys
import yaml
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from environment.thermal_env import ThermalEnv
from utils.data_logger import DataLogger
from utils.visualization import plot_training_progress
from utils.callbacks import EpisodeLoggerCallback


def train_a2c(config_path: str = None,
              total_timesteps: int = 100000,
              save_dir: str = "trained_models/a2c",
              log_dir: str = "data/logs",
              experiment_name: str = None,
              seed: int = 42):
    """
    Train A2C agent.
    
    Args:
        config_path: Path to configuration file
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        log_dir: Directory for logs
        experiment_name: Name of experiment
        seed: Random seed
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"a2c_{timestamp}"
    
    print("=" * 70)
    print(f"Training A2C Agent: {experiment_name}")
    print("=" * 70)
    
    # Load config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'thermal_config.yaml'
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override total timesteps if specified
    if total_timesteps != config['training']['total_timesteps']:
        config['training']['total_timesteps'] = total_timesteps
    
    a2c_config = config['training']['a2c']
    
    # Create environments
    print("\nCreating training environment...")
    train_env = ThermalEnv(config_path=config_path, random_weather=True)
    train_env = Monitor(train_env)
    train_env.reset(seed=seed)
    
    print("Creating evaluation environment...")
    eval_env = ThermalEnv(config_path=config_path, random_weather=True)
    eval_env = Monitor(eval_env)
    eval_env.reset(seed=seed + 1)
    
    # Initialize data logger
    logger = DataLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        log_steps=False
    )
    logger.save_config({
        'algorithm': 'A2C',
        'total_timesteps': total_timesteps,
        'seed': seed,
        **a2c_config
    })
    print(f"DataLogger initialized: {logger.get_experiment_dir()}")
    
    # Create A2C model
    print("\nInitializing A2C model...")
    model = A2C(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=a2c_config['learning_rate'],
        n_steps=a2c_config['n_steps'],
        gamma=a2c_config['gamma'],
        gae_lambda=a2c_config['gae_lambda'],
        ent_coef=a2c_config['ent_coef'],
        vf_coef=a2c_config['vf_coef'],
        max_grad_norm=a2c_config['max_grad_norm'],
        policy_kwargs=a2c_config['policy_kwargs'],
        tensorboard_log=os.path.join(config['logging']['tensorboard_dir'], experiment_name),
        verbose=config['training']['verbose'],
        seed=seed,
    )
    
    print(f"\nModel parameters:")
    print(f"  Policy: MlpPolicy")
    print(f"  Learning rate: {a2c_config['learning_rate']}")
    print(f"  n_steps: {a2c_config['n_steps']}")
    print(f"  Entropy coefficient: {a2c_config['ent_coef']}")
    print(f"  Network architecture: {a2c_config['policy_kwargs']['net_arch']}")
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, experiment_name),
        log_path=logger.get_experiment_dir(),
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=os.path.join(save_dir, experiment_name),
        name_prefix='a2c_model',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Episode logger for visualization
    episode_logger_callback = EpisodeLoggerCallback(logger, verbose=0)
    
    callback_list = CallbackList([eval_callback, checkpoint_callback, episode_logger_callback])
    
    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("-" * 70)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=config['training']['log_interval'],
            progress_bar=True
        )
        
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
        
        # Save final model
        final_model_path = os.path.join(save_dir, experiment_name, "a2c_final")
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Generate training report
        print("\nGenerating training report...")
        import pandas as pd
        episodes_df = pd.read_csv(os.path.join(logger.get_experiment_dir(), 'episodes.csv'))
        os.makedirs(os.path.join(logger.get_experiment_dir(), 'plots'), exist_ok=True)
        plot_training_progress(
            episodes_df=episodes_df,
            save_path=os.path.join(logger.get_experiment_dir(), 'plots', 'training_progress.png')
        )
        print(f"Training progress plot saved to {logger.get_experiment_dir()}/plots/training_progress.png")
        print(f"✓ Training report created in {logger.get_experiment_dir()}/plots")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current model...")
        interrupt_path = os.path.join(save_dir, experiment_name, "a2c_interrupted")
        model.save(interrupt_path)
        print(f"Model saved to: {interrupt_path}")
    
    print(f"\nExperiment results saved to: {logger.get_experiment_dir()}")
    
    return model, logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train A2C agent for heat pump control')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    train_a2c(
        config_path=args.config,
        total_timesteps=args.timesteps,
        experiment_name=args.name,
        seed=args.seed
    )
