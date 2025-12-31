"""
Performance Metrics and Evaluation

Functions for calculating and comparing RL agent performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def calculate_episode_metrics(step_data: pd.DataFrame,
                              T_comfort_min: float = 20.0,
                              T_comfort_max: float = 22.0,
                              dt_hours: float = 0.25) -> Dict:
    """
    Calculate comprehensive metrics from episode data.
    
    Args:
        step_data: DataFrame with step-by-step episode data
        T_comfort_min: Minimum comfortable temperature
        T_comfort_max: Maximum comfortable temperature
        dt_hours: Time step duration in hours
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['episode_length'] = len(step_data)
    metrics['total_reward'] = step_data['reward'].sum() if 'reward' in step_data else 0
    
    # Temperature metrics
    if 'T_indoor' in step_data:
        metrics['avg_temperature'] = step_data['T_indoor'].mean()
        metrics['min_temperature'] = step_data['T_indoor'].min()
        metrics['max_temperature'] = step_data['T_indoor'].max()
        metrics['std_temperature'] = step_data['T_indoor'].std()
        
        # Comfort metrics
        too_cold = (step_data['T_indoor'] < T_comfort_min).sum()
        too_hot = (step_data['T_indoor'] > T_comfort_max).sum()
        metrics['comfort_violations'] = too_cold + too_hot
        metrics['comfort_violation_pct'] = (metrics['comfort_violations'] / len(step_data)) * 100
        metrics['hours_too_cold'] = too_cold * dt_hours
        metrics['hours_too_hot'] = too_hot * dt_hours
        
        # Thermal discomfort integral (degree-hours)
        discomfort = 0
        for _, row in step_data.iterrows():
            T = row['T_indoor']
            if T < T_comfort_min:
                discomfort += (T_comfort_min - T) * dt_hours
            elif T > T_comfort_max:
                discomfort += (T - T_comfort_max) * dt_hours
        metrics['thermal_discomfort_deg_hours'] = discomfort
    
    # Energy metrics
    if 'P_electrical' in step_data:
        # Convert W to kWh
        metrics['total_energy_kwh'] = (step_data['P_electrical'].sum() * dt_hours) / 1000
        metrics['avg_power_kw'] = step_data['P_electrical'].mean() / 1000
        metrics['max_power_kw'] = step_data['P_electrical'].max() / 1000
        metrics['energy_per_step_kwh'] = metrics['total_energy_kwh'] / len(step_data)
    
    # Heat pump metrics
    if 'COP' in step_data:
        # Only calculate for non-zero COP (when heat pump is on)
        active_steps = step_data[step_data['COP'] > 0]
        if len(active_steps) > 0:
            metrics['avg_cop'] = active_steps['COP'].mean()
            metrics['min_cop'] = active_steps['COP'].min()
            metrics['max_cop'] = active_steps['COP'].max()
        else:
            metrics['avg_cop'] = 0
            metrics['min_cop'] = 0
            metrics['max_cop'] = 0
    
    # Action distribution
    if 'action' in step_data:
        action_counts = step_data['action'].value_counts()
        for action in range(4):
            count = action_counts.get(action, 0)
            metrics[f'action_{action}_count'] = count
            metrics[f'action_{action}_pct'] = (count / len(step_data)) * 100
    
    # Thermal output metrics
    if 'Q_thermal' in step_data:
        metrics['total_thermal_kwh'] = (step_data['Q_thermal'].sum() * dt_hours) / 1000
        metrics['avg_thermal_kw'] = step_data['Q_thermal'].mean() / 1000
    
    return metrics


def calculate_cost_benefit(energy_kwh: float,
                           comfort_violations: int,
                           episode_length: int,
                           electricity_price: float = 0.25,
                           comfort_violation_cost: float = 1.0) -> Dict:
    """
    Calculate cost-benefit metrics.
    
    Args:
        energy_kwh: Total energy consumed (kWh)
        comfort_violations: Number of comfort violations
        episode_length: Number of steps
        electricity_price: Price per kWh (€)
        comfort_violation_cost: Cost per violation (€)
        
    Returns:
        Dictionary with cost metrics
    """
    energy_cost = energy_kwh * electricity_price
    comfort_cost = comfort_violations * comfort_violation_cost
    total_cost = energy_cost + comfort_cost
    
    return {
        'energy_cost_eur': energy_cost,
        'comfort_cost_eur': comfort_cost,
        'total_cost_eur': total_cost,
        'cost_per_step': total_cost / episode_length if episode_length > 0 else 0,
    }


def compare_agents(results: Dict[str, pd.DataFrame],
                  metrics_to_compare: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare multiple agents' performance.
    
    Args:
        results: Dictionary mapping agent names to their episode data
        metrics_to_compare: List of metric names to compare
        
    Returns:
        DataFrame with comparison statistics
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            'total_reward',
            'total_energy_kwh',
            'comfort_violations',
            'avg_cop',
            'thermal_discomfort_deg_hours',
        ]
    
    comparison = []
    
    for agent_name, data in results.items():
        agent_stats = {'agent': agent_name}
        
        for metric in metrics_to_compare:
            if metric in data.columns:
                agent_stats[f'{metric}_mean'] = data[metric].mean()
                agent_stats[f'{metric}_std'] = data[metric].std()
                agent_stats[f'{metric}_min'] = data[metric].min()
                agent_stats[f'{metric}_max'] = data[metric].max()
        
        comparison.append(agent_stats)
    
    return pd.DataFrame(comparison)


def calculate_baseline_performance(step_data: pd.DataFrame,
                                   T_setpoint: float = 21.0,
                                   hysteresis: float = 1.0) -> Dict:
    """
    Calculate performance of a simple rule-based baseline controller.
    
    Baseline: ON when T < setpoint - hysteresis, OFF when T > setpoint + hysteresis
    
    Args:
        step_data: DataFrame with temperature and outdoor conditions
        T_setpoint: Target temperature
        hysteresis: Hysteresis band
        
    Returns:
        Dictionary with baseline metrics
    """
    # This is a simplified calculation - would need full simulation for accuracy
    # Here we estimate based on actual temperatures
    
    T_indoor = step_data['T_indoor'].values
    actions_would_be = []
    
    for T in T_indoor:
        if T < T_setpoint - hysteresis:
            actions_would_be.append(3)  # HIGH
        elif T > T_setpoint + hysteresis:
            actions_would_be.append(0)  # OFF
        else:
            actions_would_be.append(1)  # Keep previous or LOW
    
    # Estimate energy (rough)
    action_to_power = {0: 0, 1: 2000, 2: 4000, 3: 6000}
    estimated_power = [action_to_power[a] for a in actions_would_be]
    estimated_energy = sum(estimated_power) * 0.25 / 1000  # kWh
    
    return {
        'estimated_energy_kwh': estimated_energy,
        'controller_type': 'rule_based_hysteresis',
        'setpoint': T_setpoint,
        'hysteresis': hysteresis,
    }


def calculate_efficiency_score(total_reward: float,
                               total_energy_kwh: float,
                               comfort_violations: int,
                               episode_length: int) -> float:
    """
    Calculate normalized efficiency score (0-100).
    
    Combines comfort and energy efficiency into single metric.
    Higher is better.
    
    Args:
        total_reward: Total episode reward
        total_energy_kwh: Total energy consumed
        comfort_violations: Number of comfort violations
        episode_length: Number of steps
        
    Returns:
        Efficiency score (0-100)
    """
    # Normalize components
    comfort_score = max(0, 100 - (comfort_violations / episode_length * 100))
    
    # Energy score (lower is better, normalized assuming max ~50kWh for 48h)
    energy_score = max(0, 100 - (total_energy_kwh / 50 * 100))
    
    # Weighted combination
    efficiency = 0.6 * comfort_score + 0.4 * energy_score
    
    return np.clip(efficiency, 0, 100)


def get_summary_statistics(episodes_df: pd.DataFrame) -> Dict:
    """
    Get summary statistics from multiple episodes.
    
    Args:
        episodes_df: DataFrame with episode data
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    numeric_cols = episodes_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        summary[f'{col}_mean'] = episodes_df[col].mean()
        summary[f'{col}_std'] = episodes_df[col].std()
        summary[f'{col}_median'] = episodes_df[col].median()
        summary[f'{col}_min'] = episodes_df[col].min()
        summary[f'{col}_max'] = episodes_df[col].max()
    
    summary['n_episodes'] = len(episodes_df)
    
    return summary


if __name__ == "__main__":
    """Test metrics calculation."""
    
    print("Testing Metrics Calculation")
    print("=" * 60)
    
    # Create fake episode data
    n_steps = 192
    step_data = pd.DataFrame({
        'step': range(n_steps),
        'T_indoor': 20 + np.random.randn(n_steps) * 0.5,
        'T_outdoor': 5 + np.random.randn(n_steps) * 2,
        'action': np.random.randint(0, 4, n_steps),
        'P_electrical': np.random.uniform(0, 6000, n_steps),
        'Q_thermal': np.random.uniform(0, 20000, n_steps),
        'COP': np.random.uniform(2.5, 4.5, n_steps),
        'reward': np.random.uniform(-50, 0, n_steps),
    })
    
    # Calculate metrics
    metrics = calculate_episode_metrics(step_data)
    
    print("Episode Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nCost-Benefit Analysis:")
    cost_benefit = calculate_cost_benefit(
        energy_kwh=metrics['total_energy_kwh'],
        comfort_violations=metrics['comfort_violations'],
        episode_length=n_steps
    )
    for key, value in cost_benefit.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nEfficiency Score:")
    efficiency = calculate_efficiency_score(
        total_reward=metrics['total_reward'],
        total_energy_kwh=metrics['total_energy_kwh'],
        comfort_violations=metrics['comfort_violations'],
        episode_length=n_steps
    )
    print(f"  Score: {efficiency:.1f}/100")
    
    print("\n✓ Metrics test completed!")
