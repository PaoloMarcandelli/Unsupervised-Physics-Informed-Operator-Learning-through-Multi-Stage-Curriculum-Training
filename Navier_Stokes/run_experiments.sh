#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# =============================================================================
# Raccolta di tutti i comandi per lanciare i vari esperimenti tramite
# train_model.py.  Va eseguito (o copiato riga per riga) dalla directory
# Navier_Stokes/, dove si trova train_model.py.
#
# Struttura argomenti:
#   --exp_dir   navier-stokes-vorticity | kolmogorov
#   --model     phisfno | pino | unet
#   --training  multistage | multistage_nr | singlestage | supervised
#
# Iperparametri comuni modificabili:
#   --ntrain 16   --ntest 2
#   --modes  12   --width 20    --bsize 2
#   --epochs 100  --ss_epochs 200  --sup_epochs 300
#   --nu 1e-3     --offset 40   --seed 0
#   --S 64        --T_in 10     --T 10
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. NAVIER-STOKES  ν = 1e-3  (offset=40)
# ─────────────────────────────────────────────────────────────────────────────

# --- PhIS-FNO ----------------------------------------------------------------

# MS: multi-stage con reset Adam  (proposta principale del paper)
python train_model.py --exp_dir navier-stokes-vorticity \
    --model phisfno --training multistage \
    --nu 1e-3 --offset 40 --seed 0

# MS (n-r): multi-stage senza reset Adam  (ablazione)
python train_model.py --exp_dir navier-stokes-vorticity \
    --model phisfno --training multistage_nr \
    --nu 1e-3 --offset 40 --seed 0

# SS: single-stage  (baseline PINN)
python train_model.py --exp_dir navier-stokes-vorticity \
    --model phisfno --training singlestage \
    --nu 1e-3 --offset 40 --seed 0 --ss_epochs 200

# Supervised: upper bound
python train_model.py --exp_dir navier-stokes-vorticity \
    --model phisfno --training supervised \
    --nu 1e-3 --offset 40 --seed 0 --sup_epochs 300

# --- PINO --------------------------------------------------------------------

python train_model.py --exp_dir navier-stokes-vorticity \
    --model pino --training multistage \
    --nu 1e-3 --offset 40 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model pino --training multistage_nr \
    --nu 1e-3 --offset 40 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model pino --training singlestage \
    --nu 1e-3 --offset 40 --seed 0 --ss_epochs 200

python train_model.py --exp_dir navier-stokes-vorticity \
    --model pino --training supervised \
    --nu 1e-3 --offset 40 --seed 0 --sup_epochs 300

# --- UNet --------------------------------------------------------------------

python train_model.py --exp_dir navier-stokes-vorticity \
    --model unet --training multistage \
    --nu 1e-3 --offset 40 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model unet --training multistage_nr \
    --nu 1e-3 --offset 40 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model unet --training singlestage \
    --nu 1e-3 --offset 40 --seed 0 --ss_epochs 200

python train_model.py --exp_dir navier-stokes-vorticity \
    --model unet --training supervised \
    --nu 1e-3 --offset 40 --seed 0 --sup_epochs 300

# --- ResNet-18 ---------------------------------------------------------------

python train_model.py --exp_dir navier-stokes-vorticity \
    --model resnet --training multistage \
    --nu 1e-3 --offset 40 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model resnet --training multistage_nr \
    --nu 1e-3 --offset 40 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model resnet --training singlestage \
    --nu 1e-3 --offset 40 --seed 0 --ss_epochs 200

python train_model.py --exp_dir navier-stokes-vorticity \
    --model resnet --training supervised \
    --nu 1e-3 --offset 40 --seed 0 --sup_epochs 300

# --- TF-Net-18 ---------------------------------------------------------------


python train_model.py --exp_dir navier-stokes-vorticity \
    --model tfnet --training multistage \
    --nu 1e-3 --offset 40 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model tfnet --training multistage_nr \
    --nu 1e-3 --offset 40 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model tfnet --training singlestage \
    --nu 1e-3 --offset 40 --seed 0 --ss_epochs 200

python train_model.py --exp_dir navier-stokes-vorticity \
    --model tfnet --training supervised \
    --nu 1e-3 --offset 40 --seed 0 --sup_epochs 300


# ─────────────────────────────────────────────────────────────────────────────
# 2. NAVIER-STOKES  ν = 1e-4  (offset=20)
# ─────────────────────────────────────────────────────────────────────────────

python train_model.py --exp_dir navier-stokes-vorticity \
    --model phisfno --training multistage \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model phisfno --training multistage_nr \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model phisfno --training singlestage \
    --nu 1e-4 --offset 20 --seed 0 --ss_epochs 200

python train_model.py --exp_dir navier-stokes-vorticity \
    --model phisfno --training supervised \
    --nu 1e-4 --offset 20 --seed 0 --sup_epochs 300

python train_model.py --exp_dir navier-stokes-vorticity \
    --model pino --training multistage \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model pino --training multistage_nr \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model pino --training singlestage \
    --nu 1e-4 --offset 20 --seed 0 --ss_epochs 200

python train_model.py --exp_dir navier-stokes-vorticity \
    --model pino --training supervised \
    --nu 1e-4 --offset 20 --seed 0 --sup_epochs 300

python train_model.py --exp_dir navier-stokes-vorticity \
    --model unet --training multistage \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model unet --training multistage_nr \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model unet --training singlestage \
    --nu 1e-4 --offset 20 --seed 0 --ss_epochs 200

python train_model.py --exp_dir navier-stokes-vorticity \
    --model unet --training supervised \
    --nu 1e-4 --offset 20 --seed 0 --sup_epochs 300

python train_model.py --exp_dir navier-stokes-vorticity \
    --model resnet --training multistage \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model resnet --training multistage_nr \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model resnet --training singlestage \
    --nu 1e-4 --offset 20 --seed 0 --ss_epochs 200

python train_model.py --exp_dir navier-stokes-vorticity \
    --model resnet --training supervised \
    --nu 1e-4 --offset 20 --seed 0 --sup_epochs 300

python train_model.py --exp_dir navier-stokes-vorticity \
    --model tfnet --training multistage \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model tfnet --training multistage_nr \
    --nu 1e-4 --offset 20 --seed 0

python train_model.py --exp_dir navier-stokes-vorticity \
    --model tfnet --training singlestage \
    --nu 1e-4 --offset 20 --seed 0 --ss_epochs 200

python train_model.py --exp_dir navier-stokes-vorticity \
    --model tfnet --training supervised \
    --nu 1e-4 --offset 20 --seed 0 --sup_epochs 300


# ─────────────────────────────────────────────────────────────────────────────
# 3. KOLMOGOROV  ν = 2e-3  (offset=30)
# ─────────────────────────────────────────────────────────────────────────────

python train_model.py --exp_dir kolmogorov \
    --model phisfno --training multistage \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model phisfno --training multistage_nr \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model phisfno --training singlestage \
    --nu 2e-3 --offset 30 --seed 0 --ss_epochs 200

python train_model.py --exp_dir kolmogorov \
    --model phisfno --training supervised \
    --nu 2e-3 --offset 30 --seed 0 --sup_epochs 300

python train_model.py --exp_dir kolmogorov \
    --model pino --training multistage \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model pino --training multistage_nr \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model pino --training singlestage \
    --nu 2e-3 --offset 30 --seed 0 --ss_epochs 200

python train_model.py --exp_dir kolmogorov \
    --model pino --training supervised \
    --nu 2e-3 --offset 30 --seed 0 --sup_epochs 300

python train_model.py --exp_dir kolmogorov \
    --model unet --training multistage \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model unet --training multistage_nr \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model unet --training singlestage \
    --nu 2e-3 --offset 30 --seed 0 --ss_epochs 200

python train_model.py --exp_dir kolmogorov \
    --model unet --training supervised \
    --nu 2e-3 --offset 30 --seed 0 --sup_epochs 300

python train_model.py --exp_dir kolmogorov \
    --model resnet --training multistage \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model resnet --training multistage_nr \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model resnet --training singlestage \
    --nu 2e-3 --offset 30 --seed 0 --ss_epochs 200

python train_model.py --exp_dir kolmogorov \
    --model resnet --training supervised \
    --nu 2e-3 --offset 30 --seed 0 --sup_epochs 300

python train_model.py --exp_dir kolmogorov \
    --model tfnet --training multistage \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model tfnet --training multistage_nr \
    --nu 2e-3 --offset 30 --seed 0

python train_model.py --exp_dir kolmogorov \
    --model tfnet --training singlestage \
    --nu 2e-3 --offset 30 --seed 0 --ss_epochs 200

python train_model.py --exp_dir kolmogorov \
    --model tfnet --training supervised \
    --nu 2e-3 --offset 30 --seed 0 --sup_epochs 300


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-SEED ABLATION  (seeds 11, 22, 33 — seed 0 is in section 1 above)
#    All models × all training strategies on NS ν=1e-3
#    Output: navier-stokes-vorticity/ablation/loss/ and ablation/model/
# ─────────────────────────────────────────────────────────────────────────────

for SEED in 11 22 33; do
    for MODEL in phisfno pino unet resnet tfnet; do
        for TRAINING in multistage multistage_nr singlestage supervised; do
            python navier-stokes-vorticity/ablation/train_model_ablation.py \
                --model $MODEL --training $TRAINING --seed $SEED
        done
    done
done


# # ─────────────────────────────────────────────────────────────────────────────
# # 5. ESEMPI CON IPERPARAMETRI PERSONALIZZATI
# # ─────────────────────────────────────────────────────────────────────────────

# # Più modi di Fourier e width maggiore
# python train_model.py --exp_dir navier-stokes-vorticity \
#     --model phisfno --training multistage \
#     --nu 1e-3 --offset 40 --seed 0 \
#     --modes 20 --width 32 --epochs 150

# # Più campioni di training
# python train_model.py --exp_dir navier-stokes-vorticity \
#     --model phisfno --training multistage \
#     --nu 1e-3 --offset 40 --seed 0 \
#     --ntrain 32 --bsize 4 --epochs 100

# ─────────────────────────────────────────────────────────────────────────────
# 5. KELVIN-HELMHOLTZ  (supervised baseline, all architectures)
#
# Prerequisites:
#   cd kelvinhelmotz
#   python finitevolume.py --nsamples 200 --nworkers 4 \
#       --out ../kelvin-helmholtz/simulation_dataset.mat --plot
#   cd ..
#
# Dataset:  (200, 64, 64, 20)  — sub=2 downsamples 128→64
# Offset=0, T_in=10, T=10      — first 10 snapshots → predict next 10
# nu is unused in supervised mode but appears in the run-name directory
# ─────────────────────────────────────────────────────────────────────────────

# Dataset layout: (200, 128, 128, 20), sub=2 → S=64, offset=0, T_in=10, T=10
# nu=0.0  → no viscous term in physics residual (compressible Euler)
# Residual:
#   spline models (phisfno, unet)  → ∂ρ/∂t + v_spline·∇ρ = 0
#   spectral models (pino, resnet, tfnet) → ∂ρ/∂t + u_BiotSavart·∇ρ = 0

# --- Supervised (upper bound) ------------------------------------------------
python train_model.py --exp_dir kelvin-helmholtz \
    --model phisfno --training supervised \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --sup_epochs 150

python train_model.py --exp_dir kelvin-helmholtz \
    --model pino --training supervised \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --sup_epochs 150

python train_model.py --exp_dir kelvin-helmholtz \
    --model unet --training supervised \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --sup_epochs 150

python train_model.py --exp_dir kelvin-helmholtz \
    --model resnet --training supervised \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --sup_epochs 150

python train_model.py --exp_dir kelvin-helmholtz \
    --model tfnet --training supervised \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --sup_epochs 150

# --- Multi-stage with Adam reset (main method) --------------------------------
python train_model.py --exp_dir kelvin-helmholtz \
    --model phisfno --training multistage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

python train_model.py --exp_dir kelvin-helmholtz \
    --model pino --training multistage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

python train_model.py --exp_dir kelvin-helmholtz \
    --model unet --training multistage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

python train_model.py --exp_dir kelvin-helmholtz \
    --model resnet --training multistage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

python train_model.py --exp_dir kelvin-helmholtz \
    --model tfnet --training multistage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

# --- Multi-stage without Adam reset (ablation) --------------------------------
python train_model.py --exp_dir kelvin-helmholtz \
    --model phisfno --training multistage_nr \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

python train_model.py --exp_dir kelvin-helmholtz \
    --model pino --training multistage_nr \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

python train_model.py --exp_dir kelvin-helmholtz \
    --model unet --training multistage_nr \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

python train_model.py --exp_dir kelvin-helmholtz \
    --model resnet --training multistage_nr \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

python train_model.py --exp_dir kelvin-helmholtz \
    --model tfnet --training multistage_nr \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --epochs 100

# --- Single-stage (baseline PINN) --------------------------------------------
python train_model.py --exp_dir kelvin-helmholtz \
    --model phisfno --training singlestage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --ss_epochs 200

python train_model.py --exp_dir kelvin-helmholtz \
    --model pino --training singlestage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --ss_epochs 200

python train_model.py --exp_dir kelvin-helmholtz \
    --model unet --training singlestage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --ss_epochs 200

python train_model.py --exp_dir kelvin-helmholtz \
    --model resnet --training singlestage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --ss_epochs 200

python train_model.py --exp_dir kelvin-helmholtz \
    --model tfnet --training singlestage \
    --ntrain 160 --ntest 40 --bsize 8 --sub 2 --offset 0 --nu 0.0 --seed 0 --ss_epochs 200

# ─────────────────────────────────────────────────────────────────────────────
# Plot loss comparison
# ─────────────────────────────────────────────────────────────────────────────

python plot_loss.py --exp_dir navier-stokes-vorticity --nu 1e-3 --offset 40
python plot_loss.py --exp_dir navier-stokes-vorticity --nu 1e-4 --offset 20
python plot_loss.py --exp_dir kolmogorov              --nu 2e-3 --offset 30
python plot_loss.py --exp_dir kelvin-helmholtz --nu 0.0 --offset 0 \
    --ntrain 160 --epochs 100 --ss_epochs 200 --sup_epochs 150

# ─────────────────────────────────────────────────────────────────────────────
# Kelvin-Helmholtz — visualize predictions (GIF + snapshot) for all models
# Each command runs visualize.py from within kelvin-helmholtz/ via python path.
# ─────────────────────────────────────────────────────────────────────────────

cd kelvin-helmholtz

# Output structure: animations/{model}/{training}.gif
#                   animations/{model}/{training}_snapshots.png
for MODEL in phisfno pino unet resnet tfnet; do
    for TRAINING in supervised multistage multistage_nr singlestage; do
        python visualize.py \
            --model    "$MODEL" \
            --training "$TRAINING" \
            --ntrain 160 --ntest 40 --sub 2 --offset 0 --nu 0.0 \
            --epochs 100 --ss_epochs 200 --sup_epochs 150 \
            --sample_idx 0 --fps 6
    done
done

cd ..
