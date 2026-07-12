#!/bin/bash
# run_all.sh — Esegue tutti i training del Burger 1D study.
# Modelli: phisfno (spline), pino_fd (finite diff), pino_spectral (FFT)
# Uso: bash run_all.sh
# Log:  ogni run scrive su logs/<model>_<training>.log

set -e   # esci al primo errore
mkdir -p logs

echo "======================================================"
echo " Burger 1D — full training suite"
echo " $(date)"
echo "======================================================"

# ── Helper ────────────────────────────────────────────────────────────────────
run() {
    local label="$1"; shift
    local logfile="logs/${label}.log"
    echo ""
    echo ">>> [$label] START  $(date '+%H:%M:%S')"
    python train_model.py "$@" 2>&1 | tee "$logfile"
    echo ">>> [$label] DONE   $(date '+%H:%M:%S')"
}

# ══════════════════════════════════════════════════════════════════════════════
# PhIS-FNO  (spline derivatives) — 4 seed × 4 strategie
# ══════════════════════════════════════════════════════════════════════════════

for seed in 0 11 22 33; do

    run phisfno_ms_s${seed} \
        --model phisfno --training multistage \
        --epochs 100 --ntrain 90 --ntest 10 --batch_size 10 \
        --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
        --nu 0.1 --dt 0.1 --sub 4 --k_bd 100 --seed ${seed}

    run phisfno_ms_nr_s${seed} \
        --model phisfno --training multistage_nr \
        --epochs 100 --ntrain 90 --ntest 10 --batch_size 10 \
        --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
        --nu 0.1 --dt 0.1 --sub 4 --k_bd 100 --seed ${seed}

    run phisfno_ss_s${seed} \
        --model phisfno --training singlestage \
        --ss_epochs 200 --ntrain 90 --ntest 10 --batch_size 10 \
        --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
        --nu 0.1 --dt 0.1 --sub 4 --k_bd 100 --seed ${seed}

    run phisfno_sup_s${seed} \
        --model phisfno --training supervised \
        --sup_epochs 300 --ntrain 90 --ntest 10 --batch_size 10 \
        --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
        --nu 0.1 --dt 0.1 --sub 4 --k_bd 100 --seed ${seed}

done

# ══════════════════════════════════════════════════════════════════════════════
# PINO-FD  (finite-difference derivatives)
# ══════════════════════════════════════════════════════════════════════════════

run pino_fd_ms \
    --model pino_fd --training multistage \
    --epochs 100 --ntrain 90 --ntest 10 --batch_size 10 \
    --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
    --nu 0.1 --dt 0.1 --sub 4 --k_bd 100

run pino_fd_ms_nr \
    --model pino_fd --training multistage_nr \
    --epochs 100 --ntrain 90 --ntest 10 --batch_size 10 \
    --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
    --nu 0.1 --dt 0.1 --sub 4 --k_bd 100

run pino_fd_ss \
    --model pino_fd --training singlestage \
    --ss_epochs 200 --ntrain 90 --ntest 10 --batch_size 10 \
    --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
    --nu 0.1 --dt 0.1 --sub 4 --k_bd 100

run pino_fd_sup \
    --model pino_fd --training supervised \
    --sup_epochs 300 --ntrain 90 --ntest 10 --batch_size 10 \
    --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
    --nu 0.1 --dt 0.1 --sub 4 --k_bd 100

# ══════════════════════════════════════════════════════════════════════════════
# PINO-Spectral  (FFT derivatives — expected to diverge unsupervised)
# ══════════════════════════════════════════════════════════════════════════════

run pino_spectral_ms \
    --model pino_spectral --training multistage \
    --epochs 100 --ntrain 90 --ntest 10 --batch_size 10 \
    --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
    --nu 0.1 --dt 0.1 --sub 4 --k_bd 100

run pino_spectral_ms_nr \
    --model pino_spectral --training multistage_nr \
    --epochs 100 --ntrain 90 --ntest 10 --batch_size 10 \
    --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
    --nu 0.1 --dt 0.1 --sub 4 --k_bd 100

run pino_spectral_ss \
    --model pino_spectral --training singlestage \
    --ss_epochs 200 --ntrain 90 --ntest 10 --batch_size 10 \
    --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
    --nu 0.1 --dt 0.1 --sub 4 --k_bd 100

run pino_spectral_sup \
    --model pino_spectral --training supervised \
    --sup_epochs 300 --ntrain 90 --ntest 10 --batch_size 10 \
    --modes 32 --width 64 --lr 1e-2 --sched_step 10 --sched_gamma 0.5 \
    --nu 0.1 --dt 0.1 --sub 4 --k_bd 100

# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "======================================================"
echo " All runs completed — $(date)"
echo " Logs in: logs/"
echo "======================================================"
