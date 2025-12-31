"""
Custom callbacks for training that log episode data for visualization.
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional


class EpisodeLoggerCallback(BaseCallback):
    """
    Callback that logs episode statistics to CSV for visualization.
    
    This callback extracts episode data from the Monitor wrapper and logs it
    to episodes.csv so that visualization tools can create training progress plots.
    """
    
    def __init__(self, data_logger, verbose: int = 0):
        """
        Args:
            data_logger: DataLogger instance to use for logging
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.data_logger = data_logger
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        Checks if an episode has finished and logs it.
        """
        # Check if episode finished
        if self.locals.get('dones', [False])[0]:
            # Get info from Monitor wrapper
            info = self.locals.get('infos', [{}])[0]
            
            if 'episode' in info:
                ep_info = info['episode']
                
                # Episode statistics
                episode_stats = {
                    'total_reward': ep_info['r'],
                    'episode_length': ep_info['l'],
                }
                
                # Try to extract thermal environment specific stats if available
                if 'thermal_stats' in info:
                    thermal_stats = info['thermal_stats']
                    episode_stats.update({
                        'avg_temperature': thermal_stats.get('avg_temperature', 0),
                        'min_temperature': thermal_stats.get('min_temperature', 0),
                        'max_temperature': thermal_stats.get('max_temperature', 0),
                        'comfort_violations': thermal_stats.get('comfort_violations', 0),
                        'total_energy': thermal_stats.get('total_energy_kwh', 0),
                        'avg_cop': thermal_stats.get('avg_cop', 0),
                    })
                else:
                    # Default values if thermal stats not available
                    episode_stats.update({
                        'avg_temperature': 0,
                        'min_temperature': 0,
                        'max_temperature': 0,
                        'comfort_violations': 0,
                        'total_energy': 0,
                        'avg_cop': 0,
                    })
                
                # Log to DataLogger
                self.data_logger.log_episode(
                    episode=self.episode_count,
                    episode_stats=episode_stats,
                    episode_length=int(ep_info['l']),
                    terminated=info.get('TimeLimit.truncated', False)
                )
                
                self.episode_count += 1
                
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}: reward={ep_info['r']:.1f}, length={ep_info['l']}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.verbose > 0:
            print(f"Training completed. Logged {self.episode_count} episodes.")
