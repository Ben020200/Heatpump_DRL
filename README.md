# Heat Pump Control with Deep Reinforcement Learning

A comprehensive framework for training and evaluating Deep Reinforcement Learning (DRL) agents to optimize residential heat pump control, balancing thermal comfort and energy efficiency.

## 🎯 Project Overview

This project implements a realistic thermal environment simulation based on RC (Resistance-Capacitance) network physics, integrated with multiple state-of-the-art RL algorithms:
- **SAC** (Soft Actor-Critic) - Off-policy actor-critic with maximum entropy (best performance)
- **DQN** (Deep Q-Network) - Value-based learning with experience replay
- **A2C** (Advantage Actor-Critic) - On-policy synchronous actor-critic
- **PPO** (Proximal Policy Optimization) - Policy gradient method with trust region
- **TD3** (Twin Delayed DDPG) - Off-policy continuous control

### Key Features

✅ **Physics-Based Simulation**
- 2-zone RC thermal network (air + envelope)
- Temperature-dependent heat pump COP model
- Realistic solar gains and weather patterns
- 15-minute control intervals

✅ **Multi-Objective Optimization**
- Thermal comfort (20-22°C target range)
- Energy efficiency (minimize consumption)
- Intelligent action selection (4-level discrete control)

✅ **Comprehensive Logging**
- Episode-level metrics (reward, energy, comfort)
- Step-by-step detailed logs
- TensorBoard integration
- Automated visualization

✅ **Production-Ready Code**
- Modular architecture
- Extensive documentation
- Unit tests for physics models
- Easy-to-use CLI tools

## 📊 Research Foundation

This implementation synthesizes approaches from 10 peer-reviewed studies on HVAC control with RL, focusing on:
- Thermal building models (Studies 1, 3, 9)
- Heat pump physics (Studies 5, 6)
- Model-based vs model-free RL (Studies 4, 7, 8)
- State space and reward design (Studies 1-10)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ben020200/Heatpump_DRL.git
cd Heatpump_DRL

# Install dependencies
pip install -r requirements.txt

# Generate weather data
python -m utils.weather_generator
```

### Train an Agent

```bash
# Train SAC (recommended - best performance, 26% better than PID baseline)
python agents/train_sac.py --timesteps 100000 --name my_sac_experiment

# Train DQN (value-based, good for discrete actions)
python agents/train_dqn.py --timesteps 100000 --name my_dqn_experiment

# Train A2C (on-policy, synchronous updates)
python agents/train_a2c.py --timesteps 100000 --name my_a2c_experiment

# Train PPO (stable policy gradients)
python agents/train_ppo.py --timesteps 100000 --name my_ppo_experiment

# Train TD3 (continuous control variant)
python agents/train_td3.py --timesteps 100000 --name my_td3_experiment
```

### Evaluate Models

```bash
python agents/evaluate.py \
    --models trained_models/sac/my_sac_experiment/best_model.zip \
             trained_models/dqn/my_dqn_experiment/best_model.zip \
             trained_models/a2c/my_a2c_experiment/best_model.zip \
    --names SAC DQN A2C \
    --episodes 10 \
    --output data/evaluation
```

### Quick Test

```python
from environment.thermal_env import ThermalEnv

# Create environment
env = ThermalEnv()
obs, info = env.reset()

# Run random policy
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## 📁 Project Structure

```
Heatpump_DRL/
├── PROJECT_GUIDELINE.md          # Detailed project documentation
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config/
│   └── thermal_config.yaml       # All environment parameters
├── environment/
│   ├── building_model.py         # RC thermal network
│   ├── heat_pump_model.py        # COP-based heat pump
│   └── thermal_env.py            # Gymnasium environment
├── agents/
│   ├── train_sac.py              # SAC training (recommended)
│   ├── train_dqn.py              # DQN training
│   ├── train_a2c.py              # A2C training
│   ├── train_ppo.py              # PPO training
│   ├── train_td3.py              # TD3 training
│   └── evaluate.py               # Model evaluation
├── utils/
│   ├── weather_generator.py      # Synthetic weather data
│   ├── data_logger.py            # Training/eval logging
│   ├── metrics.py                # Performance metrics
│   └── visualization.py          # Plotting utilities
├── data/
│   ├── weather/                  # Weather datasets
│   └── logs/                     # Training logs
├── trained_models/               # Saved models
└── notebooks/
    └── analysis.ipynb            # Jupyter analysis notebook
```

## 🔬 Environment Details

### State Space (9-dimensional)
```python
[
    T_indoor,              # Indoor temperature (°C)
    T_envelope,            # Envelope temperature (°C)
    T_outdoor,             # Current outdoor temp (°C)
    T_outdoor_forecast_1h, # +1h forecast (°C)
    T_outdoor_forecast_2h, # +2h forecast (°C)
    hour_sin,              # sin(2π × hour/24)
    hour_cos,              # cos(2π × hour/24)
    day_type,              # 0=weekday, 1=weekend
    previous_action        # Last action (0-3)
]
```

### Action Space (Discrete)
- **0**: OFF (0 kW)
- **1**: LOW (2 kW)
- **2**: MEDIUM (4 kW)
- **3**: HIGH (6 kW)

### Reward Function
```python
reward = comfort_penalty + energy_penalty + cycling_penalty

where:
    comfort_penalty = -10.0 × |T_indoor - 21°C|
    energy_penalty = -0.005 × P_electrical (kW)
    cycling_penalty = -0.1 × |action_t - action_{t-1}|
```

### Physics Model

**Building Dynamics:**
```
dT_indoor/dt = (1/C_air) × [Q_hp - Q_loss + Q_envelope]
dT_envelope/dt = (1/C_envelope) × [Q_solar + Q_external - Q_envelope]
```

**Heat Pump COP:**
```
COP(T_out, T_in) = COP_nominal × [1 + k1×(T_out - T_ref) - k2×(T_in - T_setpoint)]
COP ∈ [2.0, 5.0]

Q_thermal = COP × P_electrical
```

## 📈 Training Configuration

Default hyperparameters are optimized for convergence:

| Parameter | SAC | DQN | A2C | PPO | TD3 |
|-----------|-----|-----|-----|-----|-----|
| Learning Rate | 3e-4 | 1e-3 | 7e-4 | 3e-4 | 1e-3 |
| Batch Size | 256 | 64 | N/A | 64 | 100 |
| Network | [256,256] | [64,64] | [64,64] | [64,64] | [256,256] |
| Buffer Size | 50k | 50k | N/A | N/A | 50k |
| Total Timesteps | 100k | 100k | 100k | 100k | 100k |

Edit `config/thermal_config.yaml` to customize.

## 📊 Results Analysis

### Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir=runs

# View at http://localhost:6006
```

### Generate Reports

```python
from utils.visualization import create_training_report

create_training_report(
    log_dir='data/logs',
    experiment_name='my_dqn_experiment'
)
```

### Jupyter Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

## 🧪 Testing

### Test Individual Components

```bash
# Test building model
python environment/building_model.py

# Test heat pump model
python environment/heat_pump_model.py

# Test environment
python environment/thermal_env.py

# Test weather generator
python utils/weather_generator.py

# Test logger
python utils/data_logger.py

# Test metrics
python utils/metrics.py

# Test visualization
python utils/visualization.py
```

Each module has standalone tests in its `__main__` block.

## 🎯 Performance Metrics

Agents are evaluated on:

1. **Reward**: Cumulative episode reward (higher is better)
2. **Energy**: Total kWh consumed per 48-hour episode
3. **Comfort Violations**: Number of timesteps outside 20-22°C
4. **Efficiency**: Average COP achieved
5. **Stability**: Training variance (lower is better)

### Experimental Results (Final 50 Episodes)

Comparison vs. PID baseline (Kp=500, Ki=10, Kd=100):

| Algorithm | Reward | Energy (kWh) | Violations | COP | Performance vs PID |
|-----------|--------|--------------|------------|-----|-------------------|
| **SAC** | -2,152 ± 756 | 26.9 ± 18.1 | 68 ± 49 | 3.40 | **+26% better** |
| **DQN** | -3,453 ± 1,611 | 21.1 ± 12.9 | 116 ± 45 | 3.52 | -18% worse |
| **A2C** | -8,870 ± 4,025 | 4.3 ± 7.1 | 165 ± 44 | 3.66 | -204% worse |
| **PID** | -2,916 | 32.4 | 121 | - | baseline |

**Key Findings:**
- SAC achieves 26% performance improvement over PID baseline
- 17% energy reduction while improving comfort (44% fewer violations)
- Off-policy algorithms (SAC, DQN) outperform on-policy (A2C) for this task
- Maximum entropy exploration (SAC) enables superior convergence

## 🔧 Configuration

All parameters in `config/thermal_config.yaml`:

```yaml
building:
  C_air: 5.0e6              # Thermal mass (J/K)
  C_envelope: 5.0e7
  R_air_outdoor: 0.01       # Thermal resistance (K/W)
  
heat_pump:
  COP_nominal: 3.5
  power_levels:
    HIGH: 6000              # Watts
    MEDIUM: 4000
    LOW: 2000
    OFF: 0

comfort:
  T_min: 20.0               # °C
  T_max: 22.0

training:
  total_timesteps: 200000
  eval_freq: 5000
```

## 📚 Documentation

- **[PROJECT_GUIDELINE.md](PROJECT_GUIDELINE.md)**: Detailed design decisions and research synthesis
- **Code documentation**: Extensive docstrings in all modules
- **Jupyter notebook**: Interactive analysis and visualization
- **Config file**: Inline comments for all parameters

## 🤝 Contributing

This is a research project. Contributions welcome:
- Bug fixes
- Performance improvements
- New RL algorithms
- Enhanced physics models
- Documentation improvements

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Based on research from:
1. Wei et al. (2017) - Deep RL for HVAC Control
2. Yang et al. (2015) - RL for Building Energy Management
3. De Ridder et al. (2020) - Physical Models with DRL
4. Li et al. (2023) - Model-Based RL Survey
5. Kazmi et al. (2018) - RL for Heat Pumps
6. Ruelens et al. (2018) - Batch RL for Thermostat Control
7. Sutton (1991) - Dyna Architecture
8. Mohammadi et al. (2021) - Model-Based RL Control
9. Schlüter et al. (2020) - RL for HVAC in Smart Buildings
10. Zhang et al. (2022) - Hybrid RL with Predictive Models

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [Report bugs or request features](https://github.com/Ben020200/Heatpump_DRL/issues)
- Email: [Your contact info]

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{heatpump_drl,
  author = {Ben020200},
  title = {Heat Pump Control with Deep Reinforcement Learning},
  year = {2025},
  url = {https://github.com/Ben020200/Heatpump_DRL}
}
```

---

**Status**: ✅ Production Ready | 🧪 Actively Maintained | 📊 Research Quality

Built with ❤️ for sustainable building automation