"""
Weather Data Generator

Generates synthetic weather profiles for training and testing.
Includes realistic daily and seasonal patterns.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os


class WeatherGenerator:
    """
    Generate synthetic weather data with realistic patterns.
    
    Features:
        - Daily temperature cycles (diurnal variation)
        - Seasonal trends
        - Random fluctuations
        - Persistence (autocorrelation)
    """
    
    def __init__(self, 
                 T_mean: float = 5.0,
                 T_std: float = 8.0,
                 T_min: float = -15.0,
                 T_max: float = 30.0,
                 diurnal_amplitude: float = 4.0,
                 seasonal_amplitude: float = 10.0,
                 random_seed: Optional[int] = None):
        """
        Initialize weather generator.
        
        Args:
            T_mean: Mean annual temperature (°C)
            T_std: Standard deviation for random fluctuations (°C)
            T_min: Absolute minimum temperature (°C)
            T_max: Absolute maximum temperature (°C)
            diurnal_amplitude: Day-night temperature swing (°C)
            seasonal_amplitude: Winter-summer temperature difference (°C)
            random_seed: Random seed for reproducibility
        """
        self.T_mean = T_mean
        self.T_std = T_std
        self.T_min = T_min
        self.T_max = T_max
        self.diurnal_amplitude = diurnal_amplitude
        self.seasonal_amplitude = seasonal_amplitude
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def generate_episode(self, 
                        n_steps: int = 192,
                        dt_minutes: int = 15,
                        start_day: int = 0) -> np.ndarray:
        """
        Generate weather for one episode.
        
        Args:
            n_steps: Number of time steps
            dt_minutes: Time step duration in minutes
            start_day: Starting day of year (0-365)
            
        Returns:
            temperatures: Array of outdoor temperatures (°C)
        """
        # Time array (hours from episode start)
        hours = np.arange(n_steps) * (dt_minutes / 60.0)
        
        # Seasonal component (coldest on day 0, warmest on day 182)
        day_of_year = start_day + hours / 24.0
        seasonal = self.seasonal_amplitude * np.cos(2 * np.pi * (day_of_year - 182) / 365)
        
        # Diurnal component (coldest at 6am, warmest at 4pm)
        hour_of_day = hours % 24
        diurnal = self.diurnal_amplitude * np.cos(2 * np.pi * (hour_of_day - 16) / 24)
        
        # Random walk component (weather persistence)
        random_walk = np.zeros(n_steps)
        random_walk[0] = np.random.randn() * self.T_std * 0.3
        
        for i in range(1, n_steps):
            # AR(1) process with autocorrelation
            random_walk[i] = 0.95 * random_walk[i-1] + np.random.randn() * self.T_std * 0.1
        
        # Combine components
        temperatures = self.T_mean + seasonal + diurnal + random_walk
        
        # Apply physical bounds
        temperatures = np.clip(temperatures, self.T_min, self.T_max)
        
        return temperatures
    
    def generate_year(self, dt_minutes: int = 15) -> pd.DataFrame:
        """
        Generate a full year of weather data.
        
        Args:
            dt_minutes: Time step duration in minutes
            
        Returns:
            DataFrame with timestamp and temperature
        """
        steps_per_day = 24 * 60 // dt_minutes
        n_steps = 365 * steps_per_day
        
        # Generate temperatures
        temperatures = self.generate_episode(n_steps=n_steps, 
                                            dt_minutes=dt_minutes,
                                            start_day=0)
        
        # Create timestamps
        timestamps = pd.date_range(start='2025-01-01', 
                                  periods=n_steps, 
                                  freq=f'{dt_minutes}min')
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperatures,
            'hour': timestamps.hour + timestamps.minute / 60.0,
            'day_of_year': timestamps.dayofyear,
        })
        
        return df
    
    def save_year_data(self, filepath: str, dt_minutes: int = 15):
        """
        Generate and save a year of weather data.
        
        Args:
            filepath: Output file path (CSV)
            dt_minutes: Time step duration in minutes
        """
        df = self.generate_year(dt_minutes)
        df.to_csv(filepath, index=False)
        print(f"Weather data saved to {filepath}")
        print(f"  Duration: {len(df)} steps ({len(df) * dt_minutes / 60 / 24:.1f} days)")
        print(f"  Temperature range: {df['temperature'].min():.1f} to {df['temperature'].max():.1f} °C")
        print(f"  Mean temperature: {df['temperature'].mean():.1f} °C")


def create_default_weather_datasets():
    """Create default weather datasets for training and testing."""
    
    output_dir = "/workspaces/Heatpump_DRL/data/weather"
    os.makedirs(output_dir, exist_ok=True)
    
    # Winter dataset (cold weather, challenging)
    print("Generating winter weather dataset...")
    winter_gen = WeatherGenerator(
        T_mean=2.0,
        T_std=6.0,
        T_min=-15.0,
        T_max=15.0,
        diurnal_amplitude=3.0,
        seasonal_amplitude=8.0,
        random_seed=42
    )
    winter_gen.save_year_data(os.path.join(output_dir, 'winter_weather.csv'))
    
    # Mild dataset (moderate weather, balanced)
    print("\nGenerating mild weather dataset...")
    mild_gen = WeatherGenerator(
        T_mean=10.0,
        T_std=7.0,
        T_min=-5.0,
        T_max=25.0,
        diurnal_amplitude=4.0,
        seasonal_amplitude=12.0,
        random_seed=123
    )
    mild_gen.save_year_data(os.path.join(output_dir, 'mild_weather.csv'))
    
    # Test dataset (different seed for evaluation)
    print("\nGenerating test weather dataset...")
    test_gen = WeatherGenerator(
        T_mean=5.0,
        T_std=8.0,
        T_min=-10.0,
        T_max=20.0,
        diurnal_amplitude=4.0,
        seasonal_amplitude=10.0,
        random_seed=999
    )
    test_gen.save_year_data(os.path.join(output_dir, 'test_weather.csv'))
    
    print("\n✓ All weather datasets created successfully!")


if __name__ == "__main__":
    """Generate default weather datasets."""
    create_default_weather_datasets()
