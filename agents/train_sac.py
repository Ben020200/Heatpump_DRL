"""
SAC Training Script (instead of TD3)

Train a Soft Actor-Critic agent for heat pump control.
Note: SAC works better with discrete actions than TD3, which is designed for continuous actions.
"""

import os
import sys
import yaml
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
import gymnasium as gym

from environment.thermal_env import ThermalEnv
from utils.data_logger import DataLogger
from utils.visualization import create_training_report
from utils.callbacks import EpisodeLoggerCallback


class DiscreteToBoxWrapper(gym.Wrapper):
    """
    Wrapper to convert discrete action space to continuous for SAC.
    Maps discrete actions {0,1,2,3} to continuous [-1, 1] and back.
    """
    def __init__(self, env):
        super().__init__(env)
        # Convert discrete to box space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = env.observation_space
        
    def _discrete_to_continuous(self, discrete_action):
        """Map discrete action {0,1,2,3} to continuous [-1, 1]."""
        # 0 -> -1, 1 -> -0.33, 2 -> 0.33, 3 -> 1
        mapping = {0: -1.0, 1: -0.33, 2: 0.33, 3: 1.0}
        return np.array([mapping[discrete_action]], dtype=np.float32)
    
    def _continuous_to_discrete(self, continuous_action):
        """Map continuous action [-1, 1] to discrete {0,1,2,3}."""
        val = continuous_action[0]
        if val < -0.66:
            return 0  # OFF
        elif val < 0:
            return 1  # LOW
        elif val < 0.66:
            return 2  # MEDIUM
        else:
            return 3  # HIGH
    
    def step(self, action):
        discrete_action = self._continuous_to_discrete(action)
        return self.env.step(discrete_action)


def train_sac(config_path: str = None,
              total_timesteps: int = 200000,
              save_dir: str = "trained_models/sac",
              log_dir: str = "data/logs",
              experiment_name: str = None,
              seed: int = 42):
    """
    Train SAC agent.
    
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
        experiment_name = f"sac_{timestamp}"
    
    print("=" * 70)
    print(f"Training SAC Agent: {experiment_name}")
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
    
    # Use TD3 config as base (similar hyperparameters)
    sac_config = config['training']['td3'].copy()
    
    # Create environments
    print("\nCreating training environment...")
    train_env = ThermalEnv(config_path=config_path, random_weather=True)
    train_env = DiscreteToBoxWrapper(train_env)
    train_env = Monitor(train_env)
    train_env.reset(seed=seed)
    
    print("Creating evaluation environment...")
    eval_env = ThermalEnv(config_path=config_path, random_weather=True)
    eval_env = DiscreteToBoxWrapper(eval_env)
    eval_env = Monitor(eval_env)
    eval_env.reset(seed=seed + 1)
    
    # Initialize data logger
    logger = DataLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        log_steps=False
    )
    logger.save_config({
        'algorithm': 'SAC',
        'total_timesteps': total_timesteps,
        'seed': seed,
        **sac_config
    })
    
    # Create SAC model
    print("\nInitializing SAC model...")
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=sac_config['learning_rate'],
        buffer_size=sac_config['buffer_size'],
        learning_starts=sac_config['learning_starts'],
        batch_size=sac_config['batch_size'],
        tau=sac_config['tau'],
        gamma=sac_config['gamma'],
        train_freq=sac_config['train_freq'],
        gradient_steps=sac_config['gradient_steps'],
        policy_kwargs=sac_config['policy_kwargs'],
        tensorboard_log=os.path.join(config['logging']['tensorboard_dir'], experiment_name),
        verbose=config['training']['verbose'],
        seed=seed,
    )
    
    print(f"\nModel parameters:")
    print(f"  Policy: MlpPolicy")
    print(f"  Learning rate: {sac_config['learning_rate']}")
    print(f"  Buffer size: {sac_config['buffer_size']}")
    print(f"  Batch size: {sac_config['batch_size']}")
    print(f"  Network architecture: {sac_config['policy_kwargs']['net_arch']}")
    
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
        name_prefix='sac_model',
        save_replay_buffer=True,
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
        final_model_path = os.path.join(save_dir, experiment_name, "sac_final")
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Create training report
        print("\nGenerating training report...")
        create_training_report(log_dir, experiment_name)
        
        # Save summary
        logger.save_summary({
            'status': 'completed',
            'total_timesteps': total_timesteps,
            'final_model_path': final_model_path,
            'experiment_name': experiment_name,
        })
        
        print(f"\nExperiment results saved to: {logger.get_experiment_dir()}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupt_model_path = os.path.join(save_dir, experiment_name, "sac_interrupted")
        model.save(interrupt_model_path)
        print(f"Model saved to: {interrupt_model_path}")
    
    finally:
        train_env.close()
        eval_env.close()
    
    return model, logger.get_experiment_dir()


def main():
    """Command-line interface for SAC training."""
    parser = argparse.ArgumentParser(description='Train SAC agent for heat pump control')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--timesteps', type=int, default=200000, help='Total training timesteps')
    parser.add_argument('--save-dir', type=str, default='trained_models/sac', help='Model save directory')
    parser.add_argument('--log-dir', type=str, default='data/logs', help='Log directory')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_sac(
        config_path=args.config,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        experiment_name=args.name,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
