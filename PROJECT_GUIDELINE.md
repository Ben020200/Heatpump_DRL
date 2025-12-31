# Heat Pump DRL Project - Comprehensive Guideline

## Project Overview
Development of a thermal environment simulation for training Deep Reinforcement Learning agents to optimize heat pump control in residential buildings.

---

## Literature Review Summary

### Key Findings from 10 Studies

#### **Thermal Environment Modeling Approaches:**

1. **Physical Building Models (Studies 1, 3, 9)**
   - RC (Resistance-Capacitance) thermal networks
   - Heat transfer equations: Q = U·A·ΔT
   - Thermal mass considerations (walls, air, furniture)
   - Typical: 2-4 thermal zones with coupled dynamics

2. **Heat Pump Modeling (Studies 5, 6)**
   - COP (Coefficient of Performance) as function of outdoor/indoor temps
   - Power consumption: P = Q_heating / COP
   - ON/OFF or modulating control
   - Typical COP range: 2.5-4.5

3. **State Space Design (Studies 1-10)**
   - Indoor temperature (current)
   - Outdoor temperature (current + forecast)
   - Time features (hour, day type)
   - Previous actions
   - Thermal comfort violations
   - Energy price signals (where applicable)

4. **Action Space (Studies 1, 5, 6)**
   - Discrete: ON/OFF, temperature setpoints
   - Continuous: modulation level, flow rates
   - For simplicity: Discrete actions preferred initially

5. **Reward Function Design (Studies 1-10)**
   - Multi-objective: comfort + energy efficiency
   - Typical: R = -α·(comfort_penalty) - β·(energy_cost)
   - Comfort penalty: squared deviation from comfort range
   - Energy cost: kWh × price or simple power consumption

6. **Simulation Time Steps (Studies 2, 4, 8)**
   - Common: 15-30 minute intervals
   - Balances computational efficiency with control granularity
   - Episode length: 24 hours to 1 week

7. **Model-Based vs Model-Free (Studies 4, 7, 8)**
   - Model-based: Learn system dynamics, use planning
   - Model-free: Direct policy learning (PPO, DQN, TD3)
   - Hybrid approaches show promise

---

## Project Architecture

### **1. Core Components**

```
Heatpump_DRL/
├── PROJECT_GUIDELINE.md          # This file
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── config/
│   └── thermal_config.yaml       # Environment parameters
├── environment/
│   ├── __init__.py
│   ├── thermal_env.py            # Main Gym environment
│   ├── building_model.py         # RC thermal model
│   └── heat_pump_model.py        # Heat pump physics
├── agents/
│   ├── __init__.py
│   ├── train_dqn.py              # DQN training
│   ├── train_ppo.py              # PPO training
│   └── train_td3.py              # TD3 training
├── utils/
│   ├── __init__.py
│   ├── data_logger.py            # Training/evaluation logging
│   ├── visualization.py          # Plotting utilities
│   └── metrics.py                # Performance metrics
├── data/
│   ├── weather/                  # Weather data
│   └── logs/                     # Training logs
├── trained_models/               # Saved model checkpoints
└── notebooks/
    └── analysis.ipynb            # Results analysis
```

---

## Thermal Environment Specification

### **Building Model (Simplified RC Network)**

**Two-zone thermal model:**
- **Zone 1**: Indoor air (low thermal mass, responds quickly)
- **Zone 2**: Building envelope (walls, floor - high thermal mass)

**Differential equations:**
```
dT_indoor/dt = (1/C_air) * [Q_hp - Q_loss_air + Q_envelope]
dT_envelope/dt = (1/C_envelope) * [Q_solar + Q_external - Q_envelope]

where:
Q_hp = Heat pump output (control variable)
Q_loss_air = U_air * (T_indoor - T_outdoor)
Q_envelope = U_envelope * (T_envelope - T_indoor)
Q_external = U_ext * (T_outdoor - T_envelope)
Q_solar = Solar gains (time-dependent)
```

**Simplified numerical integration:**
- Method: Euler forward (simple, stable for our time steps)
- Time step: 15 minutes (900 seconds)

### **Heat Pump Model**

**COP Model (Temperature-dependent):**
```
COP(T_outdoor, T_indoor) = COP_nominal * (1 + k1*(T_outdoor - T_ref) - k2*(T_indoor - T_setpoint))
Bounded: COP ∈ [2.0, 5.0]

P_electrical = Q_heating / COP
```

**Discrete Control:**
- Actions: {OFF, LOW, MEDIUM, HIGH}
- Power levels: {0, 2kW, 4kW, 6kW}

### **State Space (Observation)**

```python
state = [
    T_indoor,              # Current indoor temperature [°C]
    T_envelope,            # Envelope temperature [°C]
    T_outdoor,             # Current outdoor temperature [°C]
    T_outdoor_forecast_1h, # +1h forecast [°C]
    T_outdoor_forecast_2h, # +2h forecast [°C]
    hour_sin,              # sin(2π * hour/24)
    hour_cos,              # cos(2π * hour/24)
    day_type,              # 0=weekday, 1=weekend
    previous_action,       # Last action taken
]
# Total: 9-dimensional continuous state
```

### **Action Space**

```python
# Discrete: 4 actions
actions = {
    0: OFF (0 kW),
    1: LOW (2 kW),
    2: MEDIUM (4 kW),
    3: HIGH (6 kW)
}
```

### **Reward Function**

```python
# Multi-objective reward
comfort_penalty = {
    0                           if T_comfort_min ≤ T_indoor ≤ T_comfort_max
    -10 * (T_comfort_min - T_indoor)²  if T_indoor < T_comfort_min
    -10 * (T_indoor - T_comfort_max)²  if T_indoor > T_comfort_max
}

energy_penalty = -0.1 * P_electrical  # €/kWh scaled

reward = comfort_penalty + energy_penalty

# Comfort range: 20-22°C (adjustable)
```

### **Episode Configuration**

- **Duration**: 48 hours (192 steps @ 15min intervals)
- **Initial conditions**: Random T_indoor ∈ [18, 24]°C
- **Weather**: Synthetic or real data (hourly interpolated)
- **Termination**: Episode completes or severe violation (T < 10°C or T > 35°C)

---

## RL Algorithms - Implementation Plan

### **1. DQN (Deep Q-Network)**
- **Architecture**: 2-layer MLP (64-64 neurons)
- **Hyperparameters**:
  - Learning rate: 1e-3
  - Batch size: 64
  - Gamma: 0.99
  - Buffer size: 50,000
  - Target update: 1000 steps
  - Exploration: ε-greedy (1.0 → 0.05)

### **2. PPO (Proximal Policy Optimization)**
- **Architecture**: Actor-Critic, 2-layer MLP (64-64)
- **Hyperparameters**:
  - Learning rate: 3e-4
  - Batch size: 64
  - n_steps: 2048
  - Gamma: 0.99
  - GAE lambda: 0.95
  - Clip range: 0.2

### **3. TD3 (Twin Delayed DDPG)**
- **Note**: Continuous action space version
- **Modification**: Use discrete wrapper or direct discretization
- **Architecture**: Actor-Critic, 2-layer MLP (256-256)
- **Hyperparameters**:
  - Learning rate: 1e-3
  - Batch size: 100
  - Gamma: 0.99
  - Tau: 0.005
  - Policy delay: 2

---

## Data Recording & Monitoring

### **Training Metrics (Logged Every Episode)**
```python
training_log = {
    'episode': int,
    'total_reward': float,
    'avg_temperature': float,
    'comfort_violations': int,
    'total_energy_kwh': float,
    'avg_cop': float,
    'episode_length': int,
    'exploration_rate': float,  # For DQN
}
```

### **Detailed State Logs (Every Step - Optional/Sampled)**
```python
step_log = {
    'timestamp': datetime,
    'step': int,
    'T_indoor': float,
    'T_outdoor': float,
    'action': int,
    'reward': float,
    'Q_hp': float,
    'P_electrical': float,
    'COP': float,
}
```

### **Visualization Outputs**
1. Training curves (reward, energy, comfort)
2. Temperature profiles (indoor vs outdoor vs setpoint)
3. Action distribution heatmaps
4. COP efficiency over time
5. Comparison plots (DQN vs PPO vs TD3)

---

## Implementation Phases

### **Phase 1: Environment Development (Priority 1)**
- [ ] Implement RC thermal model (`building_model.py`)
- [ ] Implement heat pump model (`heat_pump_model.py`)
- [ ] Create Gym environment (`thermal_env.py`)
- [ ] Write configuration file (`thermal_config.yaml`)
- [ ] Unit tests for physics models
- [ ] Validate against expected behavior (manual control)

### **Phase 2: Data Infrastructure (Priority 2)**
- [ ] Weather data loader (synthetic or real)
- [ ] Data logger implementation (`data_logger.py`)
- [ ] Metrics calculation (`metrics.py`)
- [ ] Directory structure setup

### **Phase 3: RL Agent Training (Priority 3)**
- [ ] DQN training script with SB3
- [ ] PPO training script with SB3
- [ ] TD3 training script (with discrete action handling)
- [ ] Hyperparameter configurations
- [ ] Model checkpointing

### **Phase 4: Analysis & Visualization (Priority 4)**
- [ ] Plotting utilities (`visualization.py`)
- [ ] Jupyter notebook for analysis
- [ ] Comparison framework
- [ ] Performance benchmarking

### **Phase 5: Documentation & Refinement (Priority 5)**
- [ ] README with usage instructions
- [ ] Code documentation
- [ ] Example runs
- [ ] Results interpretation guide

---

## Key Design Decisions

### **Simplifications (For Optimal Learning)**
1. **2-zone model** instead of multi-room (reduces state space)
2. **Discrete actions** for DQN/PPO compatibility
3. **Deterministic weather** within episode (reduces variance)
4. **No occupancy modeling** initially (can add later)
5. **Fixed electricity price** (can add ToU pricing later)
6. **Euler integration** (simple, stable, fast)

### **Realism Maintained**
1. **Temperature-dependent COP** (real physics)
2. **Thermal mass dynamics** (building inertia)
3. **Weather coupling** (external disturbances)
4. **Comfort-energy tradeoff** (realistic objectives)
5. **Physical constraints** (COP bounds, temp limits)

---

## Success Metrics

### **Environment Validation**
- [ ] Physics make sense (heat flows correctly)
- [ ] Stable without control (reaches equilibrium)
- [ ] Controllable (actions affect temperature)
- [ ] Repeatable (same initial conditions → same results)

### **RL Training Success**
- [ ] Agents learn (reward increases over time)
- [ ] Maintain comfort (violations < 5% of time)
- [ ] Energy efficiency (better than baseline rule-based)
- [ ] Generalization (works on unseen weather)

### **Baseline Comparison**
- **Rule-based thermostat**: ON if T < 20°C, OFF if T > 22°C
- **Target**: RL agents should outperform on comfort AND energy

---

## Technical Stack

### **Core Dependencies**
```
python >= 3.8
gymnasium >= 0.29.0
stable-baselines3 >= 2.0.0
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
pyyaml >= 6.0
tensorboard >= 2.13.0
```

### **Optional**
```
jupyter
seaborn
scikit-learn (for normalization)
```

---

## Risk Mitigation

### **Potential Issues & Solutions**

1. **Unstable training**
   - Solution: Careful reward scaling, state normalization
   
2. **Poor exploration**
   - Solution: Entropy bonuses, longer epsilon decay
   
3. **Comfort violations**
   - Solution: Increase comfort penalty weight
   
4. **Computational cost**
   - Solution: Vectorized environments, shorter episodes
   
5. **Overfitting to training weather**
   - Solution: Diverse weather scenarios, validation set

---

## Next Steps

1. Review and approve this guideline
2. Set up project structure
3. Implement Phase 1 (Environment)
4. Test environment thoroughly
5. Proceed to RL training

---

**Document Version**: 1.0  
**Date**: 2025-12-31  
**Status**: Planning Complete - Ready for Implementation
