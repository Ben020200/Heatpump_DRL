"""
Heat Pump Model - Temperature-Dependent COP

This module implements a realistic heat pump model with:
    - Temperature-dependent Coefficient of Performance (COP)
    - Discrete power level control
    - Thermal inertia for smooth operation
    
The COP model captures the key physics: heat pumps are more efficient
when the temperature difference is smaller.
"""

import numpy as np
from typing import Tuple, Dict


class HeatPumpModel:
    """
    Heat pump model with temperature-dependent COP.
    
    The COP (Coefficient of Performance) determines how efficiently
    the heat pump converts electrical energy to thermal energy:
        Q_thermal = COP × P_electrical
    
    COP decreases when:
        - Outdoor temperature is lower (harder to extract heat)
        - Indoor temperature is higher (larger temperature lift)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize heat pump model.
        
        Args:
            config: Dictionary containing heat pump parameters
        """
        # COP parameters
        self.COP_nominal = config['COP_nominal']
        self.COP_min = config['COP_min']
        self.COP_max = config['COP_max']
        
        # Temperature coefficients
        self.k1 = config['k1']  # Outdoor temp coefficient (positive)
        self.k2 = config['k2']  # Indoor temp coefficient (negative)
        self.T_outdoor_ref = config['T_outdoor_ref']
        self.T_indoor_ref = config['T_indoor_ref']
        
        # Power levels (W)
        self.power_levels = config['power_levels']
        self.action_to_power = {
            0: self.power_levels['OFF'],
            1: self.power_levels['LOW'],
            2: self.power_levels['MEDIUM'],
            3: self.power_levels['HIGH'],
        }
        
        # Thermal inertia (exponential smoothing)
        self.thermal_inertia = config.get('thermal_inertia', 0.9)
        
        # State
        self.current_thermal_output = 0.0  # Current thermal output (W)
        self.previous_action = 0
        
    def calculate_cop(self, T_outdoor: float, T_indoor: float) -> float:
        """
        Calculate temperature-dependent COP.
        
        Linearized model around reference conditions:
            COP = COP_nom * [1 + k1*(T_out - T_out_ref) - k2*(T_in - T_in_ref)]
        
        Physical interpretation:
            - Warmer outdoor: easier to extract heat → higher COP
            - Cooler indoor: smaller temperature lift → higher COP
        
        Args:
            T_outdoor: Outdoor temperature (°C)
            T_indoor: Indoor temperature (°C)
            
        Returns:
            COP: Coefficient of performance (bounded)
        """
        # Temperature deviations from reference
        dT_outdoor = T_outdoor - self.T_outdoor_ref
        dT_indoor = T_indoor - self.T_indoor_ref
        
        # Linear COP model
        cop = self.COP_nominal * (1.0 + self.k1 * dT_outdoor - self.k2 * dT_indoor)
        
        # Apply physical bounds
        cop = np.clip(cop, self.COP_min, self.COP_max)
        
        return cop
    
    def step(self, action: int, T_outdoor: float, T_indoor: float) -> Dict:
        """
        Execute heat pump action for one time step.
        
        Args:
            action: Control action (0=OFF, 1=LOW, 2=MEDIUM, 3=HIGH)
            T_outdoor: Current outdoor temperature (°C)
            T_indoor: Current indoor temperature (°C)
            
        Returns:
            Dictionary containing:
                - Q_thermal: Thermal output (W)
                - P_electrical: Electrical power consumption (W)
                - COP: Coefficient of performance
                - action: Action taken
        """
        # Get electrical power for this action
        P_electrical = self.action_to_power.get(action, 0.0)
        
        # Calculate current COP
        cop = self.calculate_cop(T_outdoor, T_indoor)
        
        # Calculate instantaneous thermal output
        Q_target = P_electrical * cop
        
        # Apply thermal inertia (exponential smoothing)
        # Prevents instantaneous jumps in thermal output
        self.current_thermal_output = (
            self.thermal_inertia * self.current_thermal_output +
            (1 - self.thermal_inertia) * Q_target
        )
        
        # Store action
        self.previous_action = action
        
        return {
            'Q_thermal': self.current_thermal_output,
            'P_electrical': P_electrical,
            'COP': cop,
            'action': action,
            'power_level': P_electrical / 1000.0,  # kW
        }
    
    def reset(self):
        """Reset heat pump state."""
        self.current_thermal_output = 0.0
        self.previous_action = 0
        
    def get_action_space_size(self) -> int:
        """Get number of discrete actions."""
        return len(self.action_to_power)
    
    def get_power_range(self) -> Tuple[float, float]:
        """Get min and max electrical power."""
        powers = list(self.action_to_power.values())
        return min(powers), max(powers)


def plot_cop_surface():
    """
    Visualize COP as function of outdoor and indoor temperatures.
    Useful for understanding heat pump performance characteristics.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Test configuration
        config = {
            'COP_nominal': 3.5,
            'COP_min': 2.0,
            'COP_max': 5.0,
            'k1': 0.03,
            'k2': 0.02,
            'T_outdoor_ref': 7.0,
            'T_indoor_ref': 21.0,
            'power_levels': {'OFF': 0, 'LOW': 2000, 'MEDIUM': 4000, 'HIGH': 6000},
            'thermal_inertia': 0.9,
        }
        
        hp = HeatPumpModel(config)
        
        # Create temperature grids
        T_outdoor = np.linspace(-15, 20, 50)
        T_indoor = np.linspace(15, 30, 50)
        T_out_grid, T_in_grid = np.meshgrid(T_outdoor, T_indoor)
        
        # Calculate COP for each combination
        COP_grid = np.zeros_like(T_out_grid)
        for i in range(T_out_grid.shape[0]):
            for j in range(T_out_grid.shape[1]):
                COP_grid[i, j] = hp.calculate_cop(T_out_grid[i, j], T_in_grid[i, j])
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 3D surface
        contour = ax1.contourf(T_out_grid, T_in_grid, COP_grid, levels=20, cmap='RdYlGn')
        ax1.set_xlabel('Outdoor Temperature (°C)')
        ax1.set_ylabel('Indoor Temperature (°C)')
        ax1.set_title('Heat Pump COP Surface')
        plt.colorbar(contour, ax=ax1, label='COP')
        
        # COP vs outdoor temp at fixed indoor temp
        T_indoor_fixed = 21.0
        cop_vs_outdoor = [hp.calculate_cop(t, T_indoor_fixed) for t in T_outdoor]
        ax2.plot(T_outdoor, cop_vs_outdoor, 'b-', linewidth=2, label=f'T_indoor = {T_indoor_fixed}°C')
        ax2.axhline(y=config['COP_nominal'], color='r', linestyle='--', label='COP_nominal')
        ax2.axhline(y=config['COP_min'], color='gray', linestyle=':', label='COP_min')
        ax2.axhline(y=config['COP_max'], color='gray', linestyle=':', label='COP_max')
        ax2.set_xlabel('Outdoor Temperature (°C)')
        ax2.set_ylabel('COP')
        ax2.set_title('COP vs Outdoor Temperature')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('/workspaces/Heatpump_DRL/data/cop_characteristics.png', dpi=150)
        print("COP characteristics plot saved to data/cop_characteristics.png")
        
    except ImportError:
        print("Matplotlib not available for plotting")


if __name__ == "__main__":
    """Test the heat pump model."""
    
    # Test configuration
    test_config = {
        'COP_nominal': 3.5,
        'COP_min': 2.0,
        'COP_max': 5.0,
        'k1': 0.03,
        'k2': 0.02,
        'T_outdoor_ref': 7.0,
        'T_indoor_ref': 21.0,
        'power_levels': {
            'OFF': 0,
            'LOW': 2000,
            'MEDIUM': 4000,
            'HIGH': 6000,
        },
        'thermal_inertia': 0.9,
    }
    
    # Create heat pump
    hp = HeatPumpModel(test_config)
    
    print("Heat Pump Model Test")
    print("=" * 50)
    print(f"Action space size: {hp.get_action_space_size()}")
    print(f"Power range: {hp.get_power_range()} W")
    print()
    
    # Test COP calculation at various conditions
    print("COP at different operating conditions:")
    print(f"{'T_outdoor':>12} {'T_indoor':>10} {'COP':>8} {'Efficiency':>12}")
    print("-" * 45)
    
    test_conditions = [
        (-10, 21),  # Very cold outdoor
        (0, 21),    # Cold outdoor
        (7, 21),    # Reference condition
        (15, 21),   # Mild outdoor
        (7, 18),    # Lower indoor setpoint
        (7, 24),    # Higher indoor setpoint
    ]
    
    for T_out, T_in in test_conditions:
        cop = hp.calculate_cop(T_out, T_in)
        efficiency = cop / test_config['COP_nominal'] * 100
        print(f"{T_out:12.1f} {T_in:10.1f} {cop:8.2f} {efficiency:11.1f}%")
    
    print()
    print("Testing heat pump operation over 10 steps:")
    print(f"{'Step':>6} {'Action':>8} {'P_elec':>10} {'Q_therm':>10} {'COP':>8}")
    print("-" * 45)
    
    hp.reset()
    T_outdoor = 5.0
    T_indoor = 20.0
    
    actions = [0, 1, 1, 2, 2, 3, 3, 2, 1, 0]  # Ramp up and down
    
    for step, action in enumerate(actions):
        result = hp.step(action, T_outdoor, T_indoor)
        print(f"{step:6d} {action:8d} {result['P_electrical']:10.1f} "
              f"{result['Q_thermal']:10.1f} {result['COP']:8.2f}")
    
    print("\nTest completed successfully!")
    print("\nGenerating COP characteristics plot...")
    plot_cop_surface()
