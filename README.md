# Unsupervised Physics-Informed Operator Learning via Multi-Stage Curriculum Training

This repository is the official implementation of the paper:

> Paolo Marcandelli, Natansh Mathur, Stefano Markidis, Martina Siena, Stefano Mariani.
> **Unsupervised Physics-Informed Operator Learning through Multi-Stage Curriculum Training.** arXiv:2602.02264 [cs.LG], 2026.
> [https://arxiv.org/abs/2602.02264](https://arxiv.org/abs/2602.02264)

```bibtex
@misc{marcandelli2026unsupervisedphysicsinformedoperatorlearning,
      title={Unsupervised Physics-Informed Operator Learning through Multi-Stage Curriculum Training},
      author={Paolo Marcandelli and Natansh Mathur and Stefano Markidis and Martina Siena and Stefano Mariani},
      year={2026},
      eprint={2602.02264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.02264},
}
```

This guide is written for someone who has never run this codebase before. It walks through, step by step, how to set up the environment and run your first experiment. If you already know the codebase, skip to [Repository structure](#repository-structure) or [Method summary](#method-summary).

---

## What this project does, in plain terms

We want to train a neural network to solve a PDE (e.g. "given an initial condition, predict how the fluid evolves") **without labeled solution data** — the network only ever sees the equation itself, not example solutions.

Training a network this way directly (all at once) tends to get stuck: early on, the network doesn't even satisfy the boundary/initial conditions, so trying to also enforce the PDE residual in the interior is like ranking a wrong answer as "close but not quite" — the optimizer has nothing reliable to climb.

The fix used here is a **curriculum**: training is split into stages.
1. In the early stages, the loss cares mostly about **matching the boundary/initial conditions**.
2. In later stages, the weight shifts towards **satisfying the PDE residual** in the interior of the domain.
3. At each stage boundary, the **optimizer (Adam) is re-initialized** — this acts like a fresh restart that helps escape the local minimum the previous stage settled into.

The proposed network architecture, **PhIS-FNO** (Physics-Informed Spline Fourier Neural Operator), predicts the coefficients of a smooth spline basis instead of raw field values. This means the derivatives needed for the PDE residual (e.g. `∂u/∂x`, `Δu`) are obtained *analytically* from the same spline coefficients the network already predicted, rather than being estimated afterwards from a noisy predicted field (which is what simpler baselines like PINO, ResNet, or TF-Net do).

The method is tested on five problems: **Poisson 2D**, **Burgers 1D**, **Navier–Stokes vorticity** (two viscosities), **Kolmogorov flow**, **cylinder wake flow**, plus a **Kelvin–Helmholtz** compressible-flow experiment.

---

## Before you start

You will need:

- **Git**, to clone this repository.
- **Conda** (Miniconda or Anaconda), to create an isolated Python environment.
- **~5 GB of free disk space** at minimum — more if you plan to generate the larger datasets (Navier–Stokes/Kolmogorov datasets can reach several GB; see [A note on disk space](#a-note-on-disk-space) below).
- A **GPU with CUDA** is strongly recommended (training is slow on CPU) but not required — everything also runs on CPU, just much slower.

You do **not** need to download any dataset by hand: every experiment includes a small script that generates its own training data from scratch (a numerical PDE solver, not a download).

---

## Step 1 — Clone the repository

```bash
git clone git@github.com:PaoloMarcandelli/Unsupervised-Physics-Informed-Operator-Learning-through-Multi-Stage-Curriculum-Training.git
cd Unsupervised-Physics-Informed-Operator-Learning-through-Multi-Stage-Curriculum-Training
```

(If you haven't set up an SSH key with GitHub, use the HTTPS URL instead: `https://github.com/PaoloMarcandelli/Unsupervised-Physics-Informed-Operator-Learning-through-Multi-Stage-Curriculum-Training.git`.)

## Step 2 — Create the Python environment

```bash
# 1) Create a new environment (Python 3.12)
conda create -n phisfno python=3.12 -y

# 2) Activate it — you should see (phisfno) at the start of your prompt afterwards
conda activate phisfno

# 3) Upgrade pip
python -m pip install --upgrade pip

# 4) Install PyTorch
#    -> If you have an NVIDIA GPU with CUDA 12.6 drivers:
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
#    -> If you don't have a GPU, install the CPU-only build instead:
#    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# 5) Install the remaining dependencies
pip install -r requirements-project.txt
```

**Check that everything installed correctly:**

```bash
python - << 'PY'
import torch, numpy as np, scipy, matplotlib
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("NumPy:", np.__version__, "SciPy:", scipy.__version__)
import matplotlib as mpl; print("Matplotlib:", mpl.__version__)
PY
```

You should see version numbers printed with no errors. `CUDA available: True` means training will use your GPU; `False` means it will fall back to CPU (slower, but it works).

---

## Step 3 — Run your first experiment (Poisson, no dataset needed)

The **Poisson 2D** experiment is the best starting point: the equation has a known closed-form solution, so there is no dataset to generate — the script builds everything on the fly.

```bash
cd Poisson
python poisson_ms.py
```

This runs the main proposed strategy: multi-stage curriculum with Adam reset between stages, followed by an L-BFGS refinement pass. While it runs, you'll see progress lines like:

```
=== Phase full_loss_1 | epochs 100 ===
Adam     1 | total 5.23e-01  res 1.10e+00  bc 2.40e-02
Adam    50 | total 1.02e-01  res 2.31e-01  bc 5.10e-03
...
```

- `total` is the full weighted loss, `res` is the PDE-residual term, `bc` is the boundary-condition term.
- When it finishes, look in the `Poisson/` folder for:
  - `poisson_eval.png` — three panels: the analytic solution, the network's prediction, and the residual error `|Δψ+f|` (lower is better).
  - `loss_curve.png` — how the loss decreased over training, with a marker showing where L-BFGS took over from Adam.
  - `loss_history.csv` — the raw numbers behind that plot.

If this ran without errors and produced `poisson_eval.png`, your environment is correctly set up and you can move on to the other experiments.

Other Poisson variants, run the same way:
```bash
python poisson_ms_nr.py   # multi-stage, but Adam is NOT reset between stages (ablation)
python poisson_ss.py      # single-stage: full loss active from epoch 0 (no curriculum)
```

---

## Step 4 — Run an experiment that needs a generated dataset (Burgers 1D)

Unlike Poisson, most experiments need a dataset first. The dataset is **generated locally** by a numerical solver script — nothing to download.

```bash
cd Burger

# Generate the dataset: 100 trajectories of viscous Burgers' equation,
# initial conditions sampled from a Gaussian Random Field, integrated with
# a pseudo-spectral solver. Takes a couple of minutes on CPU.
python dataset/burger_dataset.py --Ns 100 --Nx 8192 --nu 0.1 --out dataset/burgers1d_dataset

# Train PhIS-FNO with the main curriculum strategy
python train_model.py --model phisfno --training multistage --nu 0.1
```

While training runs, checkpoints and loss logs are written under `Burger/model/` and `Burger/loss/` (both are excluded from git — see [A note on disk space](#a-note-on-disk-space)), named after the hyperparameters used, e.g. `model/phisfno_ms_90_Adam_ep100_visc0.1_m32_w64`.

To compare against other strategies or model types, change the flags:

```bash
python train_model.py --model phisfno    --training singlestage   --nu 0.1   # PINN-style baseline, no curriculum
python train_model.py --model phisfno    --training supervised    --nu 0.1   # upper-bound baseline, trained on labels
python train_model.py --model pino_fd    --training multistage    --nu 0.1   # PINO baseline (finite-difference residual)
python train_model.py --model pino_spectral --training multistage --nu 0.1   # PINO baseline (spectral residual — shown in the paper to be less stable)
```

Full flag reference: `--data`, `--ntrain`, `--ntest`, `--modes`, `--width`, `--epochs`, `--nu`, `--dt`, `--seed` (see the top of `train_model.py` for defaults).

---

## Step 5 — Run the 2D Navier–Stokes experiments

These live under `Navier_Stokes/` and share one training script across three physical setups (`--exp_dir`): standard Navier–Stokes vorticity, Kolmogorov forced turbulence, and Kelvin–Helmholtz instability.

```bash
cd Navier_Stokes
```

**5a) Generate a dataset.** Each `--exp_dir` has its own generator, and `train_model.py` expects to find the resulting `.mat` file *inside* the matching `--exp_dir` folder:

```bash
# Navier-Stokes vorticity / Kolmogorov: pseudo-spectral solver.
# ns_2d.py has no CLI flags — open it and edit the constants inside main()
# (visc, the forcing term f, and the output filename in scipy.io.savemat(...))
# to choose which dataset you're generating, e.g.:
#   visc=1e-3, f = 0.1*(sin(2π(x+y))+cos(2π(x+y))) -> save as 'ns_data_0.001.mat'  (navier-stokes-vorticity)
#   visc=2e-3, f = -2*cos(4πy)                      -> save as 'ns_data_Kolmogorov.mat'  (kolmogorov)
python shared/ns_2d.py
mv ns_data_0.001.mat navier-stokes-vorticity/        # or ns_data_Kolmogorov.mat -> kolmogorov/

# Kelvin-Helmholtz: finite-volume solver, has proper CLI flags
cd kelvin-helmholtz
python finitevolume.py --nsamples 200 --out simulation_dataset.mat
cd ..
```

**5b) Train.** One unified entry point:

```bash
python train_model.py \
    --exp_dir navier-stokes-vorticity \
    --model phisfno \
    --training multistage \
    --nu 1e-3 --offset 40 --seed 0
```

`run_experiments.sh` in this folder lists the full command matrix (every combination of `--exp_dir`, `--model`, `--training` used in the paper) — open it and copy the line you need rather than retyping flags from scratch.

Key flags: `--exp_dir {navier-stokes-vorticity, kolmogorov, kelvin-helmholtz}`, `--model {phisfno, pino, unet, resnet, tfnet}`, `--training {multistage, multistage_nr, singlestage, supervised}`, `--ntrain`, `--modes`, `--width`, `--epochs`, `--nu`, `--S`, `--T_in`, `--T`, `--seed`.

Loss curves across models/strategies can be plotted with:
```bash
python plot_loss.py
```

---

## Step 6 — Run the cylinder wake experiment

```bash
cd "Cylinder Wake"

# Generate the dataset: 2D flow past a cylinder, finite-difference projection solver
python cylinder_wake_dataset.py --out cylinder_wake_synthetic.mat

# Train
python UNet_unsupervised.py
```

Hyperparameters (`--ntrain`, `--modes`, `--width`, `--epochs`, `--nu`, `--S`, `--T_in`, `--T`, ...) follow the same conventions as the Navier–Stokes script; see `get_params.py` for the full list and defaults. This folder isn't wired into a unified CLI yet — `UNet_unsupervised.py` is currently the only training entry point (despite the filename, it trains the same spline-based `fluid_model` architecture, not a plain UNet).

---

## A note on disk space

Two kinds of files are **deliberately excluded from this repository** (see `.gitignore`) because they are large and fully regenerable from the code:

- **Datasets** (`*.mat`) — generated by the scripts in Step 4/5/6 above. Some can be large: the full-resolution Navier–Stokes dataset is a few GB.
- **Model checkpoints** (`model/`, `*model*/` folders, `*.pt` files) — produced automatically every time you run a `train_model.py`/`poisson_*.py`/`UNet_unsupervised.py` script.

This means: after cloning, none of these exist yet — that is expected, not a bug. Running the dataset-generation command for an experiment before training it is a required step, not optional.

---

## Repository structure

```
.
├── Burger/                          # Burgers 1D — reference implementation of the method
│   ├── train_model.py               # unified entry point: --model {phisfno,pino_fd,pino_spectral}
│   │                                 #                      --training {multistage,multistage_nr,singlestage,supervised}
│   ├── models.py                    # Net1d / SimpleBlock1d / SpectralConv1d (PhIS-FNO)
│   ├── spline_models1d.py           # Hermite spline kernels + interpolate_states (1D)
│   ├── operators.py                 # autograd grad/div/laplace helpers
│   ├── utilities3.py                # MatReader, UnitGaussianNormalizer, LpLoss
│   ├── dataset/burger_dataset.py    # generates the Burgers dataset (GRF IC + ETDRK4 integrator)
│   ├── ablation_study/              # ablation variants of train_model.py + plotting
│   └── model/, loss/, image/        # checkpoints, per-phase loss CSVs, evaluation figures (gitignored)
│
├── Navier_Stokes/                   # 2D vorticity-form Navier–Stokes, 3 sub-experiments
│   ├── train_model.py               # unified entry point (see Step 5)
│   ├── run_experiments.sh           # example commands for every model/curriculum/exp_dir combo
│   ├── plot_loss.py                 # unified loss-curve plotting across phases/models
│   ├── shared/                      # code shared by all three sub-experiments below
│   │   ├── models.py                # Net2d (PhIS-FNO), fluid_model (UNet+spline), PINONet2d,
│   │   │                             # ResNet18_NS, TFNet_NS, DeepONet2D_NS_Vorticity
│   │   ├── spline_models.py         # Hermite spline kernels + interpolate_states (2D)
│   │   ├── ns_2d.py                 # pseudo-spectral NS solver — generates NS/Kolmogorov datasets
│   │   └── setups.py, operators.py, utilities3.py, random_fields.py, unet_parts.py, resolution.py
│   ├── navier-stokes-vorticity/     # ν = 1e-3 and ν = 1e-4
│   ├── kolmogorov/                  # ν = 2e-3, forced turbulence
│   └── kelvin-helmholtz/            # compressible shear-layer instability
│       ├── finitevolume.py          # Muscl-Hancock finite-volume solver — generates the dataset
│       └── visualize.py             # autoregressive rollout → 3-panel GIF
│
├── Cylinder Wake/                   # 2D flow past a cylinder — not yet unified into train_model.py
│   ├── UNet_unsupervised.py         # training script (uses fluid_model, i.e. UNet+spline)
│   ├── cylinder_wake_dataset.py     # projection-method NS solver with immersed-boundary cylinder mask
│   └── get_params.py                # CLI parser (same hyperparameter set as Navier_Stokes)
│
├── Poisson/                         # 2D Poisson — no dataset needed (closed-form solution)
│   ├── poisson_ms.py                # multi-stage, Adam reset per phase, + final L-BFGS refinement
│   ├── poisson_ms_nr.py             # multi-stage, no Adam reset
│   └── poisson_ss.py                # single-stage baseline
```

---

## Method summary

Every experiment shares the same training-loop skeleton, defined per script as a `build_phases()` function or an explicit `phases = [...]` list:

1. **Model** predicts either raw field values (`pino`, `pino_fd`, `pino_spectral`, `unet`, `resnet`, `tfnet`) or Hermite-spline coefficients (`phisfno`, decoded through `interpolate_states`, which also yields spatial derivatives analytically consistent with the prediction).
2. **Curriculum stages** ramp the PDE-residual weight `λ_res` up (and the supervision/boundary weight `λ_sup` down) across phases; spline-based models add smoothness/total-variation/mean-velocity regularizers (`λ_smooth`, `λ_tv`, `λ_mean`).
3. **Training mode** (`--training`):
   - `multistage` — re-creates the Adam optimizer (and LR scheduler) at every phase boundary. Main proposed strategy.
   - `multistage_nr` — same λ-schedule, but Adam's moment buffers persist across phases (ablation).
   - `singlestage` — the full loss is active from epoch 0 (PINN-style baseline).
   - `supervised` — pure data loss (`λ_sup=1.0, λ_res=0.0`), used as an upper-bound baseline.
   - Poisson additionally appends an **L-BFGS** refinement stage after the Adam phases.
4. **Boundary/interior masking**: the supervision loss is computed only on a thin boundary band early in training (except in `supervised` mode, which uses the full domain); the PDE-residual loss is computed on the interior only.
5. **Logging**: per-phase loss history CSVs (`loss/.../loss_<phase>.csv`), model checkpoints under `model/<tag>_<model>_<training>_...` encoding hyperparameters in the path.

---

## Troubleshooting

- **`FileNotFoundError` pointing to a `.mat` file** — you skipped the dataset-generation step for that experiment; go back to the relevant "generate the dataset" command above.
- **`CUDA out of memory`** — lower `--ntrain`/`--bsize`/`--width`, or reduce `--S` (spatial resolution).
- **Training runs but very slowly** — check `torch.cuda.is_available()` (Step 2); if `False`, you're on CPU, which is expected to be much slower for the 2D experiments.
- **`ModuleNotFoundError` for a local module like `models` or `utilities3`** — these scripts import local files by relative name and expect to be run **from inside the experiment folder** (e.g. `cd Burger` before `python train_model.py`), not from the repository root.

---

## Citation

If you use this code, please cite the paper (see the top of this file for the full BibTeX entry):

```
Marcandelli, P., Mathur, N., Markidis, S., Siena, M., & Mariani, S. (2026).
Unsupervised Physics-Informed Operator Learning through Multi-Stage Curriculum Training.
arXiv:2602.02264 [cs.LG]. https://arxiv.org/abs/2602.02264
```
