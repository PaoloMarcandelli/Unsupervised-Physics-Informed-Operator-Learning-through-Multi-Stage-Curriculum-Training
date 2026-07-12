#!/usr/bin/env bash
# =============================================================================
# run_supervised_ablation.sh
# Supervised training for all architectures × seeds 11, 22, 33 on NS nu=1e-3.
# Seed 0 is already in navier-stokes-vorticity/loss/ (main experiment).
#
# Run from the Navier_Stokes/ directory:
#   conda run -n phisfno bash run_supervised_ablation.sh
# =============================================================================

set -e

for SEED in 11 22 33; do
    for MODEL in phisfno pino unet resnet tfnet; do
        echo "======================================================="
        echo "  supervised | model=${MODEL} | seed=${SEED}"
        echo "======================================================="
        python navier-stokes-vorticity/ablation/train_model_ablation.py \
            --model "$MODEL" --training supervised --seed "$SEED"
    done
done

echo ""
echo "Done. Run seed_stats.py to see updated statistics:"
echo "  python navier-stokes-vorticity/ablation/seed_stats.py"
