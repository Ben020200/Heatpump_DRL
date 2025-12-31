# Baseline Controllers & Cost Optimization - Quick Start

## What Was Added

### 1. Three Baseline Controllers
**Location:** `baselines/controllers.py`

- **ON-OFF Controller** (Bang-Bang Thermostat)
  - Turns heat pump fully ON when T < T_min - deadband
  - Turns fully OFF when T > T_max + deadband
  - Simple but causes frequent cycling

- **PID Controller**
  - Proportional-Integral-Derivative control
  - Parameters: Kp=500, Ki=10, Kd=100
  - Smooth control but reactive (not predictive)

- **MPC Controller** (Model Predictive Control)
  - Predicts 24 steps (6 hours) ahead using physics model
  - Optimizes comfort vs cost trade-off
  - Uses weather forecasts for proactive control

### 2. Electricity Cost Tracking
**Updated:** `environment/thermal_env.py`

- Electricity price: €0.30/kWh (configurable)
- Tracked in `info['electricity_cost']` at each step
- Total episode cost in `episode_stats['total_cost']`

### 3. Updated Reward Function
Now optimizes **comfort vs cost**:

```
reward = -α*(T_violation)² - β*(P_kW * price)
```

- α (comfort_weight): 10.0 - penalizes temperature violations
- β (energy_weight): 1.0 - penalizes electricity cost
- Adjustable in `config/thermal_config.yaml`

### 4. Evaluation Framework
**Location:** `baselines/evaluate_baselines.py`

Compare all controllers on the same test episodes:
```bash
python baselines/evaluate_baselines.py --episodes 10
```

## Usage Examples

### Test Controllers Individually
```bash
python baselines/controllers.py
```

Output:
```
ON-OFF: T=19°C → 6000W, T=23°C → 0W (binary control)
PID:    T=19°C → 2000W, T=21°C → 0W (proportional)
MPC:    T=19°C → 4000W, T=22°C → 2000W (predictive)
```

### Compare All Baselines
```bash
python baselines/evaluate_baselines.py \
    --config config/thermal_config.yaml \
    --episodes 10 \
    --save-dir results/baseline_comparison
```

Generates:
- `baseline_comparison.csv` - Performance metrics table
- `baseline_comparison.png` - 4-panel comparison plot
- `temperature_trajectories.png` - Controller behavior over time

### Compare Baselines + Trained RL Agents
```bash
python baselines/evaluate_baselines.py \
    --episodes 10 \
    --rl-models models/dqn_model.zip models/ppo_model.zip \
    --rl-names DQN PPO \
    --save-dir results/all_comparison
```

## Configuration

### Adjust Comfort vs Cost Trade-off
Edit `config/thermal_config.yaml`:

```yaml
reward:
  comfort_weight: 10.0    # Higher = prioritize comfort
  energy_weight: 1.0      # Higher = prioritize cost savings

electricity:
  price_per_kwh: 0.30     # €/kWh (can vary by time/season)
```

### Tune Controller Parameters
```yaml
baselines:
  onoff:
    deadband: 0.5         # Temperature hysteresis (°C)
  
  pid:
    Kp: 500.0            # Proportional gain
    Ki: 10.0             # Integral gain
    Kd: 100.0            # Derivative gain
  
  mpc:
    horizon: 24          # Prediction steps (6 hours @ 15min)
    alpha_comfort: 10.0
    beta_cost: 1.0
    lambda_cycle: 50.0   # Penalty for action changes
```

## Action Space

**Current:** 4 discrete power levels
- 0: OFF (0W)
- 1: LOW (2000W)
- 2: MEDIUM (4000W)
- 3: HIGH (6000W)

**Adequacy:** ✅ Good for initial experiments
- Matches typical heat pump staging
- Sufficient granularity for comfort control
- Reduces RL exploration space

**To Increase Detail (Optional):**

1. **More levels** - Add to `config/thermal_config.yaml`:
```yaml
heat_pump:
  power_levels:
    'OFF': 0
    'VERY_LOW': 1000
    'LOW': 2000
    'MEDIUM_LOW': 3000
    'MEDIUM': 4000
    'MEDIUM_HIGH': 5000
    'HIGH': 6000
    'VERY_HIGH': 7000
```

2. **Continuous control** - Use SAC (already implemented) with continuous action space [0-6000]W

3. **Mode switching** - Add explicit heating/idle/defrost modes

## Expected Results

### Controller Ranking (Predicted):
1. **MPC** - Best overall (predictive + optimal)
2. **RL (PPO/SAC)** - Learns to match or beat MPC
3. **RL (DQN)** - Good but may oscillate
4. **PID** - Decent comfort, higher energy use
5. **ON-OFF** - Most cycling, comfort violations

### Metrics to Compare:
- **Mean Reward** - Higher is better
- **Energy Consumption** - Lower is better (kWh)
- **Electricity Cost** - Lower is better (€)
- **Comfort Violations** - Lower is better (# steps outside 20-22°C)
- **Mean COP** - Higher is better (efficiency)

## Next Steps

1. **Train RL agents:**
```bash
python agents/train_dqn.py --timesteps 200000 --name dqn_baseline
python agents/train_ppo.py --timesteps 200000 --name ppo_baseline
python agents/train_sac.py --timesteps 200000 --name sac_baseline
```

2. **Run comparison:**
```bash
python baselines/evaluate_baselines.py \
    --episodes 20 \
    --rl-models models/dqn_baseline.zip models/ppo_baseline.zip models/sac_baseline.zip \
    --rl-names DQN PPO SAC
```

3. **Analyze results:**
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Files Modified/Added

### New Files:
- `baselines/__init__.py` - Module initialization
- `baselines/controllers.py` - Three baseline controllers (556 lines)
- `baselines/evaluate_baselines.py` - Comparison framework (415 lines)

### Modified Files:
- `environment/thermal_env.py` - Added electricity cost tracking
- `config/thermal_config.yaml` - Added electricity pricing & baseline parameters

### Configuration Updates:
```yaml
# NEW SECTIONS:
electricity:
  price_per_kwh: 0.30

baselines:
  onoff: {...}
  pid: {...}
  mpc: {...}

reward:
  comfort_weight: 10.0
  energy_weight: 1.0
```

## Testing

All controllers tested and working:
```bash
$ python baselines/controllers.py
✓ ON-OFF: Binary control (0W or 6000W)
✓ PID: Proportional response
✓ MPC: Predictive optimization
✓ All initialized successfully!
```

## Summary

✅ **Added:** ON-OFF, PID, MPC controllers
✅ **Implemented:** Electricity cost tracking (€/kWh)
✅ **Updated:** Reward function for comfort vs cost optimization
✅ **Created:** Comprehensive evaluation framework
✅ **Validated:** All controllers working correctly

Your project now has:
- 3 classical control baselines
- 3 RL algorithms (DQN, PPO, SAC)
- Cost-aware optimization
- Complete comparison framework

**Action space (4 levels) is adequate for now** - proceed with training and compare results!
