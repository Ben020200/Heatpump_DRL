#!/bin/bash
# Monitor SAC training progress

echo "SAC Training Monitor"
echo "===================="
echo ""

while pgrep -f "train_sac.py" > /dev/null; do
    clear
    echo "SAC Training Monitor - $(date +%H:%M:%S)"
    echo "========================================"
    echo ""
    
    # Get latest progress
    PROGRESS=$(tail -100 /tmp/sac_training_full.log | grep "total_timesteps" | tail -1)
    EVAL=$(tail -100 /tmp/sac_training_full.log | grep "Eval num_timesteps" | tail -1)
    REWARD=$(tail -100 /tmp/sac_training_full.log | grep "mean_reward" | tail -1)
    
    if [ -n "$PROGRESS" ]; then
        echo "$PROGRESS"
    fi
    
    if [ -n "$REWARD" ]; then
        echo "$REWARD"
    fi
    
    if [ -n "$EVAL" ]; then
        echo ""
        echo "Latest Evaluation:"
        echo "$EVAL"
    fi
    
    echo ""
    echo "Target: 200,000 timesteps"
    echo ""
    echo "Press Ctrl+C to stop monitoring (won't stop training)"
    
    sleep 10
done

echo ""
echo "✓ SAC training complete!"
echo ""
echo "Final results:"
tail -50 /tmp/sac_training_full.log | grep -E "(Training complete|best|final)"
