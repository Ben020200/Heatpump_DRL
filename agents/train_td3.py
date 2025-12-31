"""
TD3 Training Script

Train a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent for heat pump control.
TD3 is an off-policy algorithm that works well with continuous action spaces and sparse rewards.
"""

import os
import sys
import yaml
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium import spaces
import gymnasium as gym

from environment.thermal_env import ThermalEnv
from utils.data_logger import DataLogger
from utils.visualization import plot_training_progress
from utils.callbacks import EpisodeLoggerCallback


class DiscreteToBoxWrapper(gym.Wrapper):
    """
    Wrapper to convert discrete action space to continuous for TD3.
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


def train_td3(config_path: str = None,
              total_timesteps: int = 100000,
              save_dir: str = "trained_models/td3",
              log_dir: str = "data/logs",
              experiment_name: str = None,
              seed: int = 42):
    """
    Train TD3 agent.
    
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
        experiment_name = f"td3_{timestamp}"
    
    print("=" * 70)
    print(f"Training TD3 Agent: {experiment_name}")
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
    
    td3_config = config['training']['td3']
    
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
        'algorithm': 'TD3',
        'total_timesteps': total_timesteps,
        'seed': seed,
        **td3_config
    })
    print(f"DataLogger initialized: {logger.get_experiment_dir()}")
    
    # Add action noise for exploration
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )
    
    # Create TD3 model
    print("\nInitializing TD3 model...")
    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=td3_config['learning_rate'],
        buffer_size=td3_config['buffer_size'],
        learning_starts=td3_config['learning_starts'],
        batch_size=td3_config['batch_size'],
        tau=td3_config['tau'],
        gamma=td3_config['gamma'],
        train_freq=td3_config['train_freq'],
        gradient_steps=td3_config['gradient_steps'],
        action_noise=action_noise,
        policy_delay=td3_config['policy_delay'],
        target_policy_noise=td3_config['target_policy_noise'],
        target_noise_clip=td3_config['target_noise_clip'],
        policy_kwargs=td3_config['policy_kwargs'],
        tensorboard_log=os.path.join(config['logging']['tensorboard_dir'], experiment_name),
        verbose=config['training']['verbose'],
        seed=seed,
    )
    
    print(f"\nModel parameters:")
    print(f"  Policy: MlpPolicy")
    print(f"  Learning rate: {td3_config['learning_rate']}")
    print(f"  Buffer size: {td3_config['buffer_size']}")
    print(f"  Batch size: {td3_config['batch_size']}")
    print(f"  Policy delay: {td3_config['policy_delay']}")
    print(f"  Network architecture: {td3_config['policy_kwargs']['net_arch']}")
    
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
        name_prefix='td3_model',
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
        final_model_path = os.path.join(save_dir, experiment_name, "td3_final")
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Generate training report
        print("\nGenerating training report...")
        import pandas as pd
        episodes_df = pd.read_csv(os.path.join(logger.get_experiment_dir(), 'episodes.csv'))
        plot_training_progress(
            episodes_df=episodes_df,
            save_path=os.path.join(logger.get_experiment_dir(), 'plots', 'training_progress.png')
        )
        print(f"Training progress plot saved to {logger.get_experiment_dir()}/plots/training_progress.png")
        print(f"✓ Training report created in {logger.get_experiment_dir()}/plots")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current model...")
        interrupt_path = os.path.join(save_dir, experiment_name, "td3_interrupted")
        model.save(interrupt_path)
        print(f"Model saved to: {interrupt_path}")
    
    print(f"\nExperiment results saved to: {logger.get_experiment_dir()}")
    
    return model, logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TD3 agent for heat pump control')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    train_td3(
        config_path=args.config,
        total_timesteps=args.timesteps,
        experiment_name=args.name,
        seed=args.seed
    )
