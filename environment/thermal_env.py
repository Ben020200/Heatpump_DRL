"""
Thermal Environment for Heat Pump Control

Gymnasium environment integrating building thermal model and heat pump model
for reinforcement learning training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from typing import Dict, Tuple, Optional, Any
import pandas as pd

from .building_model import BuildingThermalModel
from .heat_pump_model import HeatPumpModel


class ThermalEnv(gym.Env):
    """
    Custom Gymnasium environment for heat pump control optimization.
    
    State Space (9-dimensional):
        - T_indoor: Indoor temperature (°C)
        - T_envelope: Envelope temperature (°C)
        - T_outdoor: Current outdoor temperature (°C)
        - T_outdoor_forecast_1h: Forecast +1h (°C)
        - T_outdoor_forecast_2h: Forecast +2h (°C)
        - hour_sin: sin(2π * hour/24)
        - hour_cos: cos(2π * hour/24)
        - day_type: 0=weekday, 1=weekend
        - previous_action: Last action taken (0-3)
        
    Action Space (Discrete):
        0: OFF (0 kW)
        1: LOW (2 kW)
        2: MEDIUM (4 kW)
        3: HIGH (6 kW)
        
    Reward:
        - Comfort penalty: -α * (T_violation)² if outside comfort zone
        - Energy cost: -β * (P_electrical * electricity_price)
        - Total: reward = comfort_penalty + energy_cost
        
    Episode:
        - Duration: 48 hours (192 steps @ 15 minutes)
        - Termination: Episode completes or critical temperature violation
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 weather_data: Optional[pd.DataFrame] = None,
                 random_weather: bool = True):
        """
        Initialize thermal environment.
        
        Args:
            config_path: Path to config YAML file
            weather_data: Pre-loaded weather data (optional)
            random_weather: If True, generate random weather for each episode
        """
        super().__init__()
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'thermal_config.yaml'
            )
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize building and heat pump models
        self.building = BuildingThermalModel(self.config['building'])
        self.heat_pump = HeatPumpModel(self.config['heat_pump'])
        
        # Simulation parameters
        self.dt = self.config['simulation']['dt']
        self.episode_length = self.config['simulation']['episode_length']
        
        # Comfort settings
        self.T_comfort_min = self.config['comfort']['T_min']
        self.T_comfort_max = self.config['comfort']['T_max']
        self.T_critical_min = self.config['comfort']['T_critical_min']
        self.T_critical_max = self.config['comfort']['T_critical_max']
        self.T_initial_min = self.config['comfort']['T_initial_min']
        self.T_initial_max = self.config['comfort']['T_initial_max']
        
        # Reward configuration (support both old and new config keys)
        reward_config = self.config.get('reward', {})
        
        # Reward type: 'literature' (linear + cycling) or 'quadratic' (old squared)
        self.reward_type = reward_config.get('type', 'literature')
        
        # Setpoint for literature-based reward
        self.T_setpoint = reward_config.get('T_setpoint', 21.0)
        
        # Weights
        self.comfort_weight = reward_config.get('comfort_weight', reward_config.get('comfort_penalty_weight', 1.0))
        self.energy_weight = reward_config.get('energy_weight', reward_config.get('energy_penalty_weight', 0.02))
        self.cycling_weight = reward_config.get('cycling_weight', 0.5)
        
        # Electricity pricing (€/kWh, constant for now)
        self.electricity_price = self.config.get('electricity', {}).get('price_per_kwh', 0.30)
        
        # Weather data
        self.weather_data = weather_data
        self.random_weather = random_weather
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # {0, 1, 2, 3}
        
        # Observation space: continuous values
        # [T_indoor, T_envelope, T_outdoor, T_out_1h, T_out_2h, 
        #  sin_hour, cos_hour, day_type, prev_action]
        self.observation_space = spaces.Box(
            low=np.array([-20, -20, -20, -20, -20, -1, -1, 0, 0], dtype=np.float32),
            high=np.array([40, 40, 40, 40, 40, 1, 1, 1, 3], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.outdoor_temperatures = None
        self.previous_action = 0
        
        # Episode statistics
        self.episode_stats = {
            'total_energy': 0.0,
            'comfort_violations': 0,
            'avg_cop': 0.0,
            'avg_temperature': 0.0,
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Generate or load weather data for episode
        if self.random_weather or self.weather_data is None:
            from utils.weather_generator import WeatherGenerator
            
            weather_gen = WeatherGenerator(**self.config['simulation']['weather'])
            self.outdoor_temperatures = weather_gen.generate_episode(
                n_steps=self.episode_length + 8,  # Extra for forecasts
                dt_minutes=self.dt // 60,
                start_day=np.random.randint(0, 365)
            )
        else:
            # Sample random episode from weather data
            start_idx = np.random.randint(0, len(self.weather_data) - self.episode_length - 8)
            self.outdoor_temperatures = self.weather_data['temperature'].values[
                start_idx:start_idx + self.episode_length + 8
            ]
        
        # Random initial temperature
        T_initial = np.random.uniform(self.T_initial_min, self.T_initial_max)
        self.building.reset(T_initial)
        self.heat_pump.reset()
        self.previous_action = 0
        
        # Reset statistics
        self.episode_stats = {
            'total_energy': 0.0,
            'total_cost': 0.0,
            'comfort_violations': 0,
            'avg_cop': 0.0,
            'avg_temperature': 0.0,
            'cop_sum': 0.0,
            'temp_sum': 0.0,
            'min_temperature': T_initial,
            'max_temperature': T_initial,
        }
        
        # Reset PID-inspired reward tracking
        if hasattr(self, 'last_T_indoor'):
            del self.last_T_indoor
        if hasattr(self, 'error_accumulator'):
            self.error_accumulator = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            action: Heat pump control action (0-3)
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Current outdoor temperature
        T_outdoor = self.outdoor_temperatures[self.current_step]
        
        # Get current building state
        T_indoor, T_envelope = self.building.get_state()
        
        # Execute heat pump action
        hp_result = self.heat_pump.step(action, T_outdoor, T_indoor)
        Q_thermal = hp_result['Q_thermal']
        P_electrical = hp_result['P_electrical']
        cop = hp_result['COP']
        
        # Update building thermal state
        hour = (self.current_step * self.dt / 3600.0) % 24
        T_indoor, T_envelope, building_info = self.building.step(
            Q_hp=Q_thermal,
            T_outdoor=T_outdoor,
            hour=hour,
            dt=self.dt
        )
        
        # Calculate reward (pass action for cycling penalty if using literature reward)
        reward = self._calculate_reward(T_indoor, P_electrical, action)
        
        # Update statistics
        energy_kwh = P_electrical * (self.dt / 3600.0) / 1000.0  # kWh
        self.episode_stats['total_energy'] += energy_kwh
        self.episode_stats['total_cost'] += energy_kwh * self.electricity_price  # €
        self.episode_stats['cop_sum'] += cop
        self.episode_stats['temp_sum'] += T_indoor
        
        if not (self.T_comfort_min <= T_indoor <= self.T_comfort_max):
            self.episode_stats['comfort_violations'] += 1
        
        # Store action
        self.previous_action = action
        
        # Advance time
        self.current_step += 1
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Critical temperature violation
        if T_indoor < self.T_critical_min or T_indoor > self.T_critical_max:
            terminated = True
            reward -= 1000  # Large penalty for failure
        
        # Episode length reached
        if self.current_step >= self.episode_length:
            truncated = True
        
        # Finalize statistics if episode ended
        if terminated or truncated:
            if self.current_step > 0:
                self.episode_stats['avg_cop'] = self.episode_stats['cop_sum'] / self.current_step
                self.episode_stats['avg_temperature'] = self.episode_stats['temp_sum'] / self.current_step
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        # Calculate electricity cost for this step
        electricity_cost = (P_electrical / 1000.0) * (self.dt / 3600.0) * self.electricity_price
        
        info.update({
            'T_indoor': T_indoor,
            'T_outdoor': T_outdoor,
            'action': action,
            'P_electrical': P_electrical,
            'Q_thermal': Q_thermal,
            'COP': cop,
            'electricity_cost': electricity_cost,  # €/step
            'reward': reward,
        })
        
        # Add thermal statistics when episode ends (for logging callback)
        if terminated or truncated:
            info['thermal_stats'] = {
                'avg_temperature': self.episode_stats.get('avg_temperature', T_indoor),
                'min_temperature': self.episode_stats.get('min_temperature', T_indoor),
                'max_temperature': self.episode_stats.get('max_temperature', T_indoor),
                'comfort_violations': self.episode_stats.get('comfort_violations', 0),
                'total_energy_kwh': self.episode_stats.get('total_energy', 0),
                'avg_cop': self.episode_stats.get('avg_cop', cop),
            }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns:
            observation: 9-dimensional state vector
        """
        T_indoor, T_envelope = self.building.get_state()
        T_outdoor = self.outdoor_temperatures[self.current_step]
        
        # Forecast outdoor temperatures (+1h and +2h)
        steps_per_hour = int(3600 / self.dt)
        T_outdoor_1h = self.outdoor_temperatures[
            min(self.current_step + steps_per_hour, len(self.outdoor_temperatures) - 1)
        ]
        T_outdoor_2h = self.outdoor_temperatures[
            min(self.current_step + 2 * steps_per_hour, len(self.outdoor_temperatures) - 1)
        ]
        
        # Time features
        hour = (self.current_step * self.dt / 3600.0) % 24
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day type (simplified: weekend if day % 7 >= 5)
        day = int(self.current_step * self.dt / 86400) % 7
        day_type = 1.0 if day >= 5 else 0.0
        
        observation = np.array([
            T_indoor,
            T_envelope,
            T_outdoor,
            T_outdoor_1h,
            T_outdoor_2h,
            hour_sin,
            hour_cos,
            day_type,
            float(self.previous_action),
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, T_indoor: float, P_electrical: float, action: int = None) -> float:
        """
        Calculate reward for current state and action.
        
        Three formulations available (configured in thermal_config.yaml):
        
        1. Literature-based (type='literature'):
           reward = -α|T_indoor - T_set| - βP - λ|Δaction|
           Based on Wei et al. and other HVAC RL papers
        
        2. PID-inspired (type='pid_shaped'):
           Progressive penalty + derivative bonus + integral penalty
           Mimics PID control success through reward shaping
        
        3. Quadratic (type='quadratic', original):
           reward = -α(T_violation)² - β(P × price)
           Heavier penalty for comfort violations
        
        Args:
            T_indoor: Current indoor temperature (°C)
            P_electrical: Electrical power consumption (W)
            action: Current action (for cycling penalty)
            
        Returns:
            reward: Scalar reward value (higher is better)
        """
        
        if self.reward_type == 'pid_shaped':
            # PID-INSPIRED REWARD SHAPING
            error = abs(T_indoor - self.T_setpoint)
            
            # 1. PROPORTIONAL: Progressive comfort penalty (like PID Kp term)
            # Small errors get small linear penalty, large errors get quadratic penalty
            if error < 1.0:
                # Very close to target: small linear penalty
                comfort_penalty = -self.comfort_weight * error
            elif error < 2.0:
                # Medium error: stronger penalty
                comfort_penalty = -self.comfort_weight * (error + error**2 * 0.5)
            else:
                # Large error: heavy quadratic penalty
                comfort_penalty = -self.comfort_weight * (error**2)
            
            # 2. DERIVATIVE: Reward moving toward setpoint (like PID Kd term)
            derivative_bonus = 0.0
            if hasattr(self, 'last_T_indoor'):
                last_error = abs(self.last_T_indoor - self.T_setpoint)
                if error < last_error:
                    # Temperature moving toward setpoint - REWARD!
                    derivative_bonus = self.comfort_weight * 0.5 * (last_error - error)
                elif error > last_error:
                    # Temperature moving away from setpoint - PENALTY!
                    derivative_bonus = -self.comfort_weight * 0.5 * (error - last_error)
            self.last_T_indoor = T_indoor
            
            # 3. INTEGRAL: Penalize sustained deviations (accumulated error)
            if not hasattr(self, 'error_accumulator'):
                self.error_accumulator = 0.0
            self.error_accumulator += error
            integral_penalty = -self.comfort_weight * 0.01 * min(self.error_accumulator, 100.0)
            
            # Energy penalty (standard)
            energy_penalty = -self.energy_weight * P_electrical
            
            # Cycling penalty (smoother control)
            if action is not None and hasattr(self, 'previous_action'):
                cycling_penalty = -self.cycling_weight * abs(action - self.previous_action)
            else:
                cycling_penalty = 0.0
            
            reward = comfort_penalty + derivative_bonus + integral_penalty + energy_penalty + cycling_penalty
            
        elif self.reward_type == 'literature':
            # Literature-based reward: linear temperature deviation
            comfort_penalty = -self.comfort_weight * abs(T_indoor - self.T_setpoint)
            
            # Energy penalty: proportional to power (scaled to W)
            energy_penalty = -self.energy_weight * P_electrical
            
            # Cycling penalty: penalize action changes (smoother control)
            if action is not None and hasattr(self, 'previous_action'):
                cycling_penalty = -self.cycling_weight * abs(action - self.previous_action)
            else:
                cycling_penalty = 0.0
            
            reward = comfort_penalty + energy_penalty + cycling_penalty
            
        else:  # 'quadratic' (original implementation)
            # Comfort penalty (quadratic outside comfort zone)
            if T_indoor < self.T_comfort_min:
                comfort_penalty = -self.comfort_weight * (self.T_comfort_min - T_indoor) ** 2
            elif T_indoor > self.T_comfort_max:
                comfort_penalty = -self.comfort_weight * (T_indoor - self.T_comfort_max) ** 2
            else:
                comfort_penalty = 0.0
            
            # Energy cost (power × price)
            energy_cost = -self.energy_weight * (P_electrical / 1000.0) * self.electricity_price
            
            reward = comfort_penalty + energy_cost
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information.
        
        Returns:
            info: Dictionary with episode statistics
        """
        info = {
            'episode_step': self.current_step,
            'episode_stats': self.episode_stats.copy(),
        }
        return info
    
    def render(self):
        """Render environment (optional)."""
        if self.current_step > 0:
            T_indoor, T_envelope = self.building.get_state()
            T_outdoor = self.outdoor_temperatures[self.current_step - 1]
            print(f"Step {self.current_step}: T_indoor={T_indoor:.1f}°C, "
                  f"T_outdoor={T_outdoor:.1f}°C, Action={self.previous_action}")


# Wrapper for evaluation with detailed logging
class DetailedLoggingWrapper(gym.Wrapper):
    """
    Wrapper that logs detailed step information.
    Useful for analysis and visualization.
    """
    
    def __init__(self, env: ThermalEnv):
        super().__init__(env)
        self.step_logs = []
        
    def reset(self, **kwargs):
        self.step_logs = []
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Log detailed step information
        self.step_logs.append({
            'step': info['episode_step'],
            'T_indoor': info['T_indoor'],
            'T_outdoor': info['T_outdoor'],
            'action': info['action'],
            'P_electrical': info['P_electrical'],
            'Q_thermal': info['Q_thermal'],
            'COP': info['COP'],
            'reward': info['reward'],
        })
        
        return obs, reward, terminated, truncated, info
    
    def get_logs(self) -> pd.DataFrame:
        """Get logs as DataFrame."""
        return pd.DataFrame(self.step_logs)


if __name__ == "__main__":
    """Test the thermal environment."""
    print("Testing Thermal Environment")
    print("=" * 60)
    
    # Create environment
    env = ThermalEnv()
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation shape: {env.observation_space.shape}")
    print()
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    print()
    
    # Run a few steps with random actions
    print("Running 10 random steps:")
    print(f"{'Step':>5} {'T_in':>7} {'T_out':>7} {'Action':>7} {'Reward':>10} {'Energy':>10}")
    print("-" * 60)
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"{i+1:5d} {info['T_indoor']:7.2f} {info['T_outdoor']:7.2f} "
              f"{action:7d} {reward:10.2f} {info['P_electrical']/1000:9.2f}kW")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    print()
    print("✓ Environment test completed successfully!")
