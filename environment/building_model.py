"""
Building Thermal Model - RC Network Implementation

This module implements a simplified 2-zone RC (Resistance-Capacitance) thermal
network for residential building simulation. The model captures the essential
thermal dynamics for heat pump control optimization.

Physics Model:
    Zone 1: Indoor air (low thermal mass, fast response)
    Zone 2: Building envelope (high thermal mass, slow response)
    
    Heat flows:
        - Heat pump to indoor air
        - Indoor air to outdoor (ventilation, windows)
        - Indoor air to/from envelope
        - Envelope to outdoor (conduction through walls)
        - Solar gains to envelope
"""

import numpy as np
from typing import Tuple, Dict


class BuildingThermalModel:
    """
    Two-zone RC thermal network for building simulation.
    
    State variables:
        - T_indoor: Indoor air temperature (°C)
        - T_envelope: Building envelope temperature (°C)
    
    Inputs:
        - Q_hp: Heat pump thermal output (W)
        - T_outdoor: Outdoor air temperature (°C)
        - Q_solar: Solar heat gains (W)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize building thermal model.
        
        Args:
            config: Dictionary containing building parameters
        """
        # Thermal capacitances (J/K)
        self.C_air = float(config['C_air'])
        self.C_envelope = float(config['C_envelope'])
        
        # Thermal conductances (W/K) - inverse of resistances
        self.U_air_outdoor = 1.0 / config['R_air_outdoor']
        self.U_envelope_indoor = 1.0 / config['R_envelope_indoor']
        self.U_envelope_outdoor = 1.0 / config['R_envelope_outdoor']
        
        # Solar gains
        self.solar_gain_max = config['solar_gain_max']
        self.solar_gain_min = config['solar_gain_min']
        
        # State variables
        self.T_indoor = 20.0  # °C
        self.T_envelope = 20.0  # °C
        
        # For numerical stability
        self.dt = 900.0  # Time step in seconds (15 minutes)
        
    def reset(self, T_initial: float = 20.0) -> Tuple[float, float]:
        """
        Reset the building thermal state.
        
        Args:
            T_initial: Initial temperature for both zones (°C)
            
        Returns:
            (T_indoor, T_envelope): Initial temperatures
        """
        self.T_indoor = T_initial
        self.T_envelope = T_initial
        return self.T_indoor, self.T_envelope
    
    def calculate_solar_gain(self, hour: float) -> float:
        """
        Calculate solar heat gains based on time of day.
        
        Simplified model: sinusoidal variation with peak at solar noon.
        
        Args:
            hour: Hour of day (0-24)
            
        Returns:
            Solar heat gain (W)
        """
        # Solar gain peaks at 12:00 (solar noon), minimum at night
        # Using cosine with phase shift
        solar_factor = 0.5 * (1 + np.cos(2 * np.pi * (hour - 12) / 24))
        
        # Apply factor between min and max gains
        Q_solar = self.solar_gain_min + solar_factor * (self.solar_gain_max - self.solar_gain_min)
        
        # Zero solar gain during night (simplified)
        if hour < 6 or hour > 20:
            Q_solar *= 0.1  # Minimal gains at night
            
        return Q_solar
    
    def step(self, 
             Q_hp: float, 
             T_outdoor: float, 
             hour: float,
             dt: float = None) -> Tuple[float, float, Dict]:
        """
        Advance the thermal model by one time step using Euler integration.
        
        Heat balance equations:
            dT_indoor/dt = (1/C_air) * [Q_hp - Q_air_out + Q_envelope_in]
            dT_envelope/dt = (1/C_envelope) * [Q_solar + Q_outdoor_in - Q_envelope_in]
        
        Args:
            Q_hp: Heat pump thermal output (W)
            T_outdoor: Outdoor temperature (°C)
            hour: Current hour of day (0-24)
            dt: Time step (seconds), defaults to self.dt
            
        Returns:
            T_indoor: Updated indoor temperature (°C)
            T_envelope: Updated envelope temperature (°C)
            info: Dictionary with detailed heat flows
        """
        if dt is None:
            dt = self.dt
            
        # Calculate solar gains
        Q_solar = self.calculate_solar_gain(hour)
        
        # Heat flows (W) - positive means heat input to the zone
        
        # Indoor air heat flows
        Q_air_to_outdoor = self.U_air_outdoor * (self.T_indoor - T_outdoor)
        Q_envelope_to_indoor = self.U_envelope_indoor * (self.T_envelope - self.T_indoor)
        
        # Envelope heat flows
        Q_outdoor_to_envelope = self.U_envelope_outdoor * (T_outdoor - self.T_envelope)
        
        # Net heat flow to indoor air
        dQ_indoor = Q_hp - Q_air_to_outdoor + Q_envelope_to_indoor
        
        # Net heat flow to envelope
        dQ_envelope = Q_solar + Q_outdoor_to_envelope - Q_envelope_to_indoor
        
        # Temperature changes (Euler forward integration)
        dT_indoor = (dQ_indoor / self.C_air) * dt
        dT_envelope = (dQ_envelope / self.C_envelope) * dt
        
        # Update states
        self.T_indoor += dT_indoor
        self.T_envelope += dT_envelope
        
        # Prepare detailed info
        info = {
            'T_indoor': self.T_indoor,
            'T_envelope': self.T_envelope,
            'Q_hp': Q_hp,
            'Q_air_to_outdoor': Q_air_to_outdoor,
            'Q_envelope_to_indoor': Q_envelope_to_indoor,
            'Q_outdoor_to_envelope': Q_outdoor_to_envelope,
            'Q_solar': Q_solar,
            'dQ_indoor': dQ_indoor,
            'dQ_envelope': dQ_envelope,
            'dT_indoor_dt': dT_indoor / dt,
            'dT_envelope_dt': dT_envelope / dt,
        }
        
        return self.T_indoor, self.T_envelope, info
    
    def get_state(self) -> Tuple[float, float]:
        """
        Get current thermal state.
        
        Returns:
            (T_indoor, T_envelope): Current temperatures (°C)
        """
        return self.T_indoor, self.T_envelope
    
    def get_thermal_time_constants(self) -> Dict[str, float]:
        """
        Calculate characteristic time constants of the building.
        
        Useful for understanding thermal dynamics.
        
        Returns:
            Dictionary with time constants in hours
        """
        # Time constant = RC (thermal capacity × thermal resistance)
        tau_air = self.C_air / self.U_air_outdoor / 3600  # hours
        tau_envelope = self.C_envelope / self.U_envelope_outdoor / 3600  # hours
        
        return {
            'tau_air_hours': tau_air,
            'tau_envelope_hours': tau_envelope,
            'tau_air_steps': tau_air * 3600 / self.dt,  # in 15-min steps
            'tau_envelope_steps': tau_envelope * 3600 / self.dt,
        }


if __name__ == "__main__":
    """Test the building thermal model."""
    
    # Test configuration
    test_config = {
        'C_air': 5.0e6,
        'C_envelope': 5.0e7,
        'R_air_outdoor': 0.01,
        'R_envelope_indoor': 0.005,
        'R_envelope_outdoor': 0.02,
        'solar_gain_max': 1000.0,
        'solar_gain_min': 100.0,
    }
    
    # Create model
    model = BuildingThermalModel(test_config)
    
    # Print time constants
    print("Building Thermal Time Constants:")
    time_constants = model.get_thermal_time_constants()
    for key, value in time_constants.items():
        print(f"  {key}: {value:.2f}")
    
    # Simulate 24 hours without heating
    print("\n24-hour simulation (no heating, constant outdoor temp):")
    print(f"{'Hour':>6} {'T_indoor':>10} {'T_envelope':>12} {'Q_solar':>10}")
    print("-" * 42)
    
    model.reset(T_initial=20.0)
    T_outdoor = 5.0  # Cold outdoor temperature
    
    for step in range(96):  # 24 hours @ 15-min steps
        hour = (step * 0.25) % 24
        T_indoor, T_envelope, info = model.step(
            Q_hp=0.0,  # No heating
            T_outdoor=T_outdoor,
            hour=hour
        )
        
        if step % 4 == 0:  # Print every hour
            print(f"{hour:6.1f} {T_indoor:10.2f} {T_envelope:12.2f} {info['Q_solar']:10.1f}")
    
    print("\nTest completed successfully!")
