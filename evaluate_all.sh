#!/bin/bash
# Comprehensive evaluation of all controllers and RL agents

echo "=========================================="
echo "COMPREHENSIVE CONTROLLER EVALUATION"
echo "=========================================="
echo ""

# Wait for SAC to finish if still training
if pgrep -f "train_sac.py" > /dev/null; then
    echo "⏳ Waiting for SAC training to complete..."
    while pgrep -f "train_sac.py" > /dev/null; do
        sleep 10
    done
    echo "✓ SAC training complete!"
    echo ""
fi

# Check which models exist
echo "Checking for trained models..."
echo ""

DQN_MODEL=""
PPO_MODEL=""
SAC_MODEL=""

if [ -f "trained_models/dqn/dqn_cost_aware/best_model.zip" ]; then
    DQN_MODEL="trained_models/dqn/dqn_cost_aware/best_model.zip"
    echo "✓ Found DQN model: $DQN_MODEL"
fi

if [ -f "trained_models/ppo/ppo_cost_aware/best_model.zip" ]; then
    PPO_MODEL="trained_models/ppo/ppo_cost_aware/best_model.zip"
    echo "✓ Found PPO model: $PPO_MODEL"
fi

if [ -f "trained_models/sac/sac_cost_aware/best_model.zip" ]; then
    SAC_MODEL="trained_models/sac/sac_cost_aware/best_model.zip"
    echo "✓ Found SAC model: $SAC_MODEL"
fi

echo ""
echo "=========================================="
echo "Running comparison with 10 episodes..."
echo "=========================================="
echo ""

# Build arguments
RL_MODELS=""
RL_NAMES=""

if [ -n "$DQN_MODEL" ]; then
    RL_MODELS="$RL_MODELS $DQN_MODEL"
    RL_NAMES="$RL_NAMES DQN"
fi

if [ -n "$PPO_MODEL" ]; then
    RL_MODELS="$RL_MODELS $PPO_MODEL"
    RL_NAMES="$RL_NAMES PPO"
fi

if [ -n "$SAC_MODEL" ]; then
    RL_MODELS="$RL_MODELS $SAC_MODEL"
    RL_NAMES="$RL_NAMES SAC"
fi

# Run comparison
if [ -n "$RL_MODELS" ]; then
    python baselines/evaluate_baselines.py \
        --episodes 10 \
        --save-dir results/final_comparison \
        --rl-models $RL_MODELS \
        --rl-names $RL_NAMES
else
    echo "⚠️  No trained RL models found. Running baselines only..."
    python baselines/evaluate_baselines.py \
        --episodes 10 \
        --save-dir results/final_comparison
fi

echo ""
echo "=========================================="
echo "Results saved to: results/final_comparison/"
echo "=========================================="
echo ""
echo "View results:"
echo "  - CSV: results/final_comparison/baseline_comparison.csv"
echo "  - Plots: results/final_comparison/*.png"
echo ""
