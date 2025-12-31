"""
Baseline controllers for heat pump control comparison.

Implements three classical control strategies:
1. OnOffController (bang-bang thermostat)
2. PIDController (proportional-integral-derivative)
3. MPCController (model predictive control)

These serve as benchmarks to evaluate RL agent performance.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.building_model import BuildingThermalModel
from environment.heat_pump_model import HeatPumpModel


class BaseController:
    """Base class for all controllers."""
    
    def __init__(self, config: Dict):
        """
        Initialize controller.
        
        Args:
            config: Configuration dictionary with control parameters
        """
        self.config = config
        self.name = "Base"
        
    def select_action(self, state: np.ndarray, info: Dict) -> int:
        """
        Select control action based on current state.
        
        Args:
            state: Current state observation
            info: Additional information from environment
            
        Returns:
            Action index (0-3 for discrete actions)
        """
        raise NotImplementedError
        
    def reset(self):
        """Reset controller state (for stateful controllers like PID)."""
        pass


class OnOffController(BaseController):
    """
    Bang-bang thermostat controller.
    
    Simple two-position control:
    - Turn ON at full power when T < T_min
    - Turn OFF when T > T_max
    - Maintain previous state in deadband
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "OnOff"
        self.T_min = config['comfort']['T_min']
        self.T_max = config['comfort']['T_max']
        self.deadband = config.get('baselines', {}).get('onoff', {}).get('deadband', 0.5)
        self.last_action = 0
        
    def select_action(self, state: np.ndarray, info: Dict) -> int:
        """
        Select action based on temperature thresholds.
        
        Args:
            state: [T_indoor, T_envelope, T_outdoor, ...]
            info: Environment info dict
            
        Returns:
            0 (OFF) or 3 (FULL POWER)
        """
        T_indoor = state[0]
        
        # Below comfort zone - turn on full
        if T_indoor < self.T_min - self.deadband:
            self.last_action = 3  # FULL POWER
        # Above comfort zone - turn off
        elif T_indoor > self.T_max + self.deadband:
            self.last_action = 0  # OFF
        # In deadband - maintain previous state
        
        return self.last_action
        
    def reset(self):
        """Reset controller state."""
        self.last_action = 0


class PIDController(BaseController):
    """
    PID (Proportional-Integral-Derivative) controller.
    
    Calculates control output as:
        u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de/dt
    
    where e(t) = T_setpoint - T_indoor
    
    Output is mapped to discrete action space {0, 1, 2, 3}.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "PID"
        
        # PID gains (tuned for thermal systems with slow dynamics)
        baseline_config = config.get('baselines', {}).get('pid', {})
        self.Kp = baseline_config.get('Kp', 500.0)  # Proportional gain
        self.Ki = baseline_config.get('Ki', 10.0)   # Integral gain
        self.Kd = baseline_config.get('Kd', 100.0)  # Derivative gain
        
        # Target temperature (middle of comfort zone)
        T_min = config['comfort']['T_min']
        T_max = config['comfort']['T_max']
        self.T_setpoint = (T_min + T_max) / 2.0
        
        # PID state variables
        self.integral = 0.0
        self.last_error = 0.0
        self.dt = config['simulation']['dt'] / 3600.0  # Convert to hours
        
        # Power levels for discrete actions (convert from string keys to int keys)
        self.power_levels = {0: 0, 1: 2000, 2: 4000, 3: 6000}  # action -> watts
        self.max_power = 6000.0
        
        # Anti-windup limits for integral term
        self.integral_max = 100.0
        
    def select_action(self, state: np.ndarray, info: Dict) -> int:
        """
        Calculate PID control output and map to discrete action.
        
        Args:
            state: [T_indoor, T_envelope, T_outdoor, ...]
            info: Environment info dict
            
        Returns:
            Action index (0-3)
        """
        T_indoor = state[0]
        
        # Calculate error
        error = self.T_setpoint - T_indoor
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        I = self.Ki * self.integral
        
        # Derivative term
        d_error = (error - self.last_error) / self.dt if self.dt > 0 else 0.0
        D = self.Kd * d_error
        
        # Total control output (in Watts)
        u = P + I + D
        
        # Clip to valid power range
        u = np.clip(u, 0, self.max_power)
        
        # Map continuous output to discrete action
        action = self._map_power_to_action(u)
        
        # Update state
        self.last_error = error
        
        return action
        
    def _map_power_to_action(self, power: float) -> int:
        """
        Map continuous power value to nearest discrete action.
        
        Args:
            power: Desired power in Watts
            
        Returns:
            Action index (0-3)
        """
        # Find action with closest power level
        min_diff = float('inf')
        best_action = 0
        
        for action, power_level in self.power_levels.items():
            diff = abs(power - power_level)
            if diff < min_diff:
                min_diff = diff
                best_action = action
                
        return best_action
        
    def reset(self):
        """Reset PID state."""
        self.integral = 0.0
        self.last_error = 0.0


class MPCController(BaseController):
    """
    Model Predictive Control (MPC) for heat pump.
    
    At each timestep:
    1. Predict future states using physics model
    2. Optimize action sequence over prediction horizon
    3. Execute only the first action
    4. Repeat at next timestep
    
    Optimization objective:
        min Σ[α*(T_i - T_set)² + β*P_i + λ*cycling_i]
    
    Subject to:
        - T_min ≤ T_indoor ≤ T_max (comfort constraints)
        - 0 ≤ P_hp ≤ P_max (power constraints)
        - Physics: T(t+1) = f(T(t), P(t), weather(t))
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "MPC"
        
        # MPC parameters
        mpc_config = config.get('baselines', {}).get('mpc', {})
        self.horizon = mpc_config.get('horizon', 24)  # Prediction horizon (steps)
        self.dt = config['simulation']['dt']  # Time step in seconds
        
        # Comfort bounds
        self.T_min = config['comfort']['T_min']
        self.T_max = config['comfort']['T_max']
        self.T_setpoint = (self.T_min + self.T_max) / 2.0
        
        # Cost weights (matching reward function)
        mpc_config = config.get('baselines', {}).get('mpc', {})
        self.alpha_comfort = mpc_config.get('alpha_comfort', 10.0)
        self.beta_cost = mpc_config.get('beta_cost', 1.0)
        self.lambda_cycle = mpc_config.get('lambda_cycle', 50.0)
        
        # Power levels (W) for discrete actions
        self.power_levels = [0, 2000, 4000, 6000]  # List indexed by action
        
        # Initialize physics models for prediction
        self.building_model = BuildingThermalModel(config['building'])
        self.hp_model = HeatPumpModel(config['heat_pump'])
        
        # Store last action for cycling penalty
        self.last_action = 0
        
    def select_action(self, state: np.ndarray, info: Dict) -> int:
        """
        Solve MPC optimization problem and return first action.
        
        Args:
            state: [T_indoor, T_envelope, T_outdoor, solar, hour_sin, hour_cos, day_sin, day_cos, prev_action]
            info: Environment info dict with 'weather_forecast'
            
        Returns:
            Optimal action index (0-3)
        """
        # Extract current state
        T_indoor = state[0]
        T_envelope = state[1]
        T_outdoor = state[2]
        
        # Get weather forecast from info
        weather_forecast = info.get('weather_forecast', None)
        
        if weather_forecast is None:
            # Fallback: assume constant weather
            weather_forecast = [(T_outdoor, state[3])] * self.horizon
        
        # Solve optimization problem
        optimal_actions = self._optimize_trajectory(
            T_indoor, T_envelope, weather_forecast
        )
        
        # Return first action
        action = optimal_actions[0]
        self.last_action = action
        
        return action
        
    def _optimize_trajectory(
        self, 
        T_indoor_0: float, 
        T_envelope_0: float,
        weather_forecast: list
    ) -> list:
        """
        Solve MPC optimization over prediction horizon.
        
        Uses simplified discrete optimization (evaluating all action sequences
        would be 4^24 ≈ 280 trillion combinations, so we use greedy approximation).
        
        Args:
            T_indoor_0: Initial indoor temperature
            T_envelope_0: Initial envelope temperature
            weather_forecast: List of (T_outdoor, solar_gain) tuples
            
        Returns:
            List of optimal actions
        """
        # For computational efficiency, use receding horizon greedy approach
        # Alternative: use scipy.optimize for continuous relaxation
        
        optimal_actions = []
        best_cost = float('inf')
        
        # Greedy optimization: at each step, choose action minimizing immediate + predicted cost
        for first_action in range(4):
            # Simulate trajectory with this first action
            T_indoor = T_indoor_0
            T_envelope = T_envelope_0
            total_cost = 0.0
            actions = [first_action]
            
            for t in range(min(self.horizon, len(weather_forecast))):
                # Current action
                if t == 0:
                    action = first_action
                else:
                    # Greedy: choose action minimizing immediate cost
                    action = self._greedy_action(T_indoor, self.T_setpoint)
                    actions.append(action)
                
                # Get weather
                T_outdoor, solar_gain = weather_forecast[t]
                
                # Simulate one step
                power = self.power_levels[action]
                hp_result = self.hp_model.step(action, T_outdoor, T_indoor)
                Q_hp = hp_result['Q_thermal']
                
                # Update building state
                T_indoor_new, T_envelope_new = self._simulate_building_step(
                    T_indoor, T_envelope, T_outdoor, Q_hp, solar_gain
                )
                
                # Calculate cost
                cost = self._calculate_cost(T_indoor, power, action, actions[t-1] if t > 0 else self.last_action)
                total_cost += cost * (0.95 ** t)  # Discount factor
                
                # Update state
                T_indoor = T_indoor_new
                T_envelope = T_envelope_new
            
            # Track best first action
            if total_cost < best_cost:
                best_cost = total_cost
                optimal_actions = actions
        
        return optimal_actions
        
    def _greedy_action(self, T_indoor: float, T_setpoint: float) -> int:
        """
        Greedy action selection for horizon beyond first step.
        
        Args:
            T_indoor: Current indoor temperature
            T_setpoint: Target temperature
            
        Returns:
            Action index (0-3)
        """
        # Simple heuristic: proportional to temperature error
        error = T_setpoint - T_indoor
        
        if error > 1.0:
            return 3  # Full power
        elif error > 0.5:
            return 2  # Medium power
        elif error > 0.0:
            return 1  # Low power
        else:
            return 0  # Off
            
    def _simulate_building_step(
        self, 
        T_indoor: float, 
        T_envelope: float,
        T_outdoor: float,
        Q_hp: float,
        solar_gain: float
    ) -> Tuple[float, float]:
        """
        Simulate building thermal dynamics one step forward.
        
        Args:
            T_indoor: Indoor air temperature
            T_envelope: Building envelope temperature
            T_outdoor: Outdoor temperature
            Q_hp: Heat pump thermal output (W)
            solar_gain: Solar heat gain (W)
            
        Returns:
            (T_indoor_new, T_envelope_new)
        """
        # Heat flows using building model conductances
        Q_air_outdoor = self.building_model.U_air_outdoor * (T_indoor - T_outdoor)
        Q_envelope_indoor = self.building_model.U_envelope_indoor * (T_envelope - T_indoor)
        Q_envelope_outdoor = self.building_model.U_envelope_outdoor * (T_envelope - T_outdoor)
        
        # Temperature changes (Euler forward integration)
        dT_indoor = (Q_hp + Q_envelope_indoor - Q_air_outdoor + solar_gain) / self.building_model.C_air * self.dt
        dT_envelope = (-Q_envelope_indoor - Q_envelope_outdoor) / self.building_model.C_envelope * self.dt
        
        T_indoor_new = T_indoor + dT_indoor
        T_envelope_new = T_envelope + dT_envelope
        
        return T_indoor_new, T_envelope_new
        
    def _calculate_cost(self, T_indoor: float, power: float, action: int, prev_action: int) -> float:
        """
        Calculate instantaneous cost for MPC objective.
        
        Args:
            T_indoor: Indoor temperature
            power: Heat pump power consumption
            action: Current action
            prev_action: Previous action
            
        Returns:
            Cost value
        """
        # Comfort cost (quadratic penalty outside comfort zone)
        if T_indoor < self.T_min:
            comfort_cost = self.alpha_comfort * (self.T_min - T_indoor) ** 2
        elif T_indoor > self.T_max:
            comfort_cost = self.alpha_comfort * (T_indoor - self.T_max) ** 2
        else:
            comfort_cost = 0.0
        
        # Energy cost
        energy_cost = self.beta_cost * power
        
        # Cycling penalty (penalize action changes)
        cycling_cost = self.lambda_cycle if action != prev_action else 0.0
        
        return comfort_cost + energy_cost + cycling_cost
        
    def reset(self):
        """Reset controller state."""
        self.last_action = 0
        self.building_model.reset()
        self.hp_model.reset()


# Test controllers
if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'thermal_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("BASELINE CONTROLLERS TEST")
    print("="*60)
    
    # Test state (typical winter conditions)
    state = np.array([20.5, 21.0, -2.0, 500.0, 0.5, 0.866, 0.0, 1.0, 0])
    info = {'weather_forecast': [(-2.0, 500.0)] * 24}
    
    # Power levels mapping
    power_levels = {0: 0, 1: 2000, 2: 4000, 3: 6000}
    
    # Test OnOff controller
    print("\n1. ON-OFF CONTROLLER")
    print("-" * 40)
    onoff = OnOffController(config)
    for T in [19.0, 20.0, 21.0, 22.0, 23.0]:
        test_state = state.copy()
        test_state[0] = T
        action = onoff.select_action(test_state, info)
        print(f"  T_indoor = {T:.1f}°C → Action = {action} ({power_levels[action]}W)")
    
    # Test PID controller
    print("\n2. PID CONTROLLER")
    print("-" * 40)
    pid = PIDController(config)
    for T in [19.0, 20.0, 21.0, 22.0, 23.0]:
        test_state = state.copy()
        test_state[0] = T
        action = pid.select_action(test_state, info)
        print(f"  T_indoor = {T:.1f}°C → Action = {action} ({power_levels[action]}W)")
    
    # Test MPC controller
    print("\n3. MPC CONTROLLER")
    print("-" * 40)
    mpc = MPCController(config)
    for T in [19.0, 20.0, 21.0, 22.0, 23.0]:
        test_state = state.copy()
        test_state[0] = T
        action = mpc.select_action(test_state, info)
        print(f"  T_indoor = {T:.1f}°C → Action = {action} ({power_levels[action]}W)")
    
    print("\n" + "="*60)
    print("All controllers initialized successfully!")
    print("="*60)
