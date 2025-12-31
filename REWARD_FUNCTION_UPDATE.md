# Reward Function Update - Literature-Based Implementation

## Problem Identified

All three RL agents (DQN, PPO, SAC) failed to learn effective control policies:
- **DQN**: Learned to minimize cost by rarely heating (underheating)
- **PPO**: Learned catastrophic overheating (100% power → 35°C → early termination)
- **SAC**: Learned to never heat at all (0 kWh, massive comfort violations)

Root cause: **Reward function design incompatible with RL learning**

## Original (Broken) Reward Function

```python
# Quadratic comfort penalty with comfort range
if T < 20°C:
    comfort_penalty = -10 × (20 - T)²
elif T > 22°C:
    comfort_penalty = -10 × (T - 22)²
else:
    comfort_penalty = 0  # Dead zone!

energy_penalty = -1.0 × (P_kW × €0.30)

reward = comfort_penalty + energy_penalty
```

### Issues:

1. **Dead Zone Problem**: Zero penalty inside 20-22°C range creates sparse feedback
   - Agent doesn't learn to maintain specific temperature
   - No gradient to guide policy toward optimal point

2. **Quadratic Penalty**: Heavily punishes deviations
   - Small violations get huge penalties
   - Discourages exploration near boundaries
   - Creates unstable learning dynamics

3. **No Smoothness Penalty**: Nothing prevents rapid on/off cycling
   - Agents learn extreme strategies (always on or always off)
   - No incentive for gradual control

4. **Scale Mismatch**: Comfort weight (10) vs Energy weight (1.0 × 0.30)
   - Imbalanced optimization makes learning difficult
   - Agents focus on one objective at expense of other

## Literature-Based Reward Function (from PDF)

Based on **Wei et al. (2017)** and the project PDF:

```python
reward = -α|T_indoor - T_setpoint| - βP_consumed - λ|action_t - action_t-1|
```

Where:
- **α** (comfort_weight): Weight for temperature deviation (linear)
- **β** (energy_weight): Weight for power consumption
- **λ** (cycling_weight): Weight for action changes (smoothness)

### Advantages:

1. **Continuous Feedback**: Always provides gradient toward setpoint
   - No dead zone - agent always knows how far from optimal
   - Linear penalty easier to learn than quadratic

2. **Specific Target**: Targets T_setpoint = 21°C
   - Clear objective instead of vague 20-22°C range
   - More stable learning signal

3. **Cycling Penalty**: Penalizes |Δaction|
   - Encourages smooth, gradual control
   - Prevents rapid on/off switching
   - Extends compressor lifespan

4. **Better Scaling**: Simpler units make tuning easier
   - α = 1.0 (1 reward unit per 1°C deviation)
   - β = 0.02 (power in Watts, scaled to match temperature units)
   - λ = 0.5 (moderate penalty for action changes)

## New Configuration

```yaml
# config/thermal_config.yaml
reward:
  type: 'literature'        # or 'quadratic' for old behavior
  T_setpoint: 21.0          # Target temperature (°C)
  comfort_weight: 1.0       # α - temperature deviation weight
  energy_weight: 0.02       # β - power consumption weight (W → reward scale)
  cycling_weight: 0.5       # λ - action change penalty
```

## Implementation Changes

### thermal_env.py Updates:

1. **Added reward type selection**:
   - `reward_type = 'literature'` or `'quadratic'`
   - Maintains backward compatibility

2. **New parameters**:
   - `T_setpoint`: Target temperature for literature reward
   - `cycling_weight`: Penalty for action changes

3. **Updated `_calculate_reward()` method**:
   ```python
   if self.reward_type == 'literature':
       comfort = -comfort_weight × |T_indoor - T_setpoint|
       energy = -energy_weight × P_electrical
       cycling = -cycling_weight × |action - previous_action|
       reward = comfort + energy + cycling
   ```

## Expected Improvements

### Learning Stability:
- **Smoother gradients**: Linear penalty provides clearer learning signal
- **Continuous feedback**: No dead zone, always knows direction to improve
- **Balanced objectives**: Better weight scaling reduces conflicting signals

### Control Quality:
- **Temperature tracking**: Should maintain 21°C ± 0.5°C
- **Smooth operation**: Cycling penalty encourages gradual power adjustments
- **Energy efficiency**: Still optimizes energy within comfort constraints

### Training Convergence:
- **Faster learning**: Clearer reward signal accelerates convergence
- **Stable policies**: Smoother control prevents extreme strategies
- **Better exploration**: Linear penalty less punishing, encourages trying near-optimal states

## Next Steps

### 1. Train New Agents (Recommended)

```bash
# Train DQN with literature reward
python agents/train_dqn.py --timesteps 200000 --name dqn_literature

# Train PPO with literature reward
python agents/train_ppo.py --timesteps 200000 --name ppo_literature

# Train SAC with literature reward
python agents/train_sac.py --timesteps 200000 --name sac_literature
```

### 2. Compare Results

```bash
python baselines/evaluate_baselines.py \
    --episodes 10 \
    --save-dir results/literature_comparison \
    --rl-models trained_models/dqn/dqn_literature/best_model.zip \
                trained_models/ppo/ppo_literature/best_model.zip \
                trained_models/sac/sac_literature/best_model.zip \
    --rl-names DQN_lit PPO_lit SAC_lit
```

### 3. Fine-Tune Weights (If Needed)

If agents still struggle, adjust:

**More comfort focus**:
```yaml
comfort_weight: 2.0      # Increase to prioritize temperature
energy_weight: 0.01      # Decrease for less energy penalty
```

**More energy focus**:
```yaml
comfort_weight: 0.5      # Decrease to allow more deviation
energy_weight: 0.05      # Increase to save more energy
```

**Smoother control**:
```yaml
cycling_weight: 1.0      # Increase to reduce cycling
```

## References

1. **Wei et al. (2017)** - "Deep Reinforcement Learning for Building HVAC Control"
   - First to apply DQN to commercial HVAC
   - Used linear temperature deviation reward
   
2. **Project PDF (Ben Goff)** - "Model-Based Reinforcement Learning for Heat Pump Control"
   - Explicitly defines: `r = -α|T - T_set| - βP - λC_cycles`
   - Emphasizes compressor longevity through cycling penalty

3. **Kazmi et al. (2018)** - "RL for Domestic Heat Pumps"
   - Demonstrated importance of action smoothness
   - Linear rewards outperformed quadratic for heat pumps

## Success Metrics

RL agents should achieve:
- **Comfort**: < 10% time outside 20.5-21.5°C
- **Energy**: 25-35 kWh per 48h episode
- **Reward**: Better than PID's -496
- **Stability**: No early terminations, consistent performance

If successful, agents should:
- ✅ Outperform or match PID controller
- ✅ Maintain smoother control than OnOff
- ✅ Learn predictive strategies using weather forecasts
- ✅ Generalize to unseen weather patterns

---

**Status**: ✅ Implementation Complete - Ready for Training
**Date**: 2025-12-31
**Version**: Literature-Based Reward v1.0
