"""
seed_stats.py — Mean ± std of convergence plateau across seeds.

For every (model, training) combination on NS nu=1e-3, loads the test_full
loss series for seeds 0, 11, 22, 33, detects the final plateau with the same
algorithm used in plot_loss.py, and prints a summary table.

  Seed 0   → navier-stokes-vorticity/loss/<run_name>/
  Seeds 11,22,33 → navier-stokes-vorticity/ablation/loss/<run_name>_seed{seed}/

Usage (from the Navier_Stokes/ directory):
    python navier-stokes-vorticity/ablation/seed_stats.py
"""

import os
import sys

import numpy as np
import pandas as pd
from glob import glob

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))   # .../ablation
_NS_DIR  = os.path.dirname(_HERE)                        # .../navier-stokes-vorticity

MAIN_LOSS_DIR    = os.path.join(_NS_DIR, 'loss')
ABLATION_LOSS_DIR = os.path.join(_HERE,  'loss')

# ── Fixed hyperparameters (must match the trained runs) ───────────────────────
NTRAIN  = 16
NU      = 1e-3
MODES   = 12
WIDTH   = 20
T       = 10
EPOCHS      = 100
SS_EPOCHS   = 200
SUP_EPOCHS  = 300

SEEDS   = [0, 11, 22, 33]
MODELS  = ['phisfno', 'pino', 'unet', 'resnet', 'tfnet']
TRAININGS = ['multistage', 'multistage_nr', 'singlestage', 'supervised']


# ════════════════════════════════════════════════════════════════════════════
# HELPERS  (identical to plot_loss.py)
# ════════════════════════════════════════════════════════════════════════════

def final_plateau_stats(y, rel_tol=2e-3, min_len=20, tail_max=100):
    """Detect the final flat region of a loss curve and return its mean/std."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.nan, np.nan
    eps = 1e-12
    dy_rel = np.abs(np.diff(y)) / (np.abs(y[:-1]) + eps)
    cnt = 1
    for k in range(n - 2, -1, -1):
        if dy_rel[k] <= rel_tol:
            cnt += 1
        else:
            break
    length = max(cnt, min_len)
    if length < min_len:
        length = min(tail_max, n)
    i0  = n - length
    seg = y[i0:]
    return float(np.mean(seg)), float(np.std(seg))


def infer_phase_names(training, loss_dir):
    if training in ('multistage', 'multistage_nr'):
        if os.path.isfile(os.path.join(loss_dir, 'loss_boundary_only_2.csv')):
            return ['boundary_only_1', 'boundary_only_2',
                    'full_loss_1', 'full_loss_2', 'full_loss_3']
        return ['boundary_only_1', 'full_loss_1', 'full_loss_2']
    if training == 'singlestage':
        return ['full_loss_1']
    if training == 'supervised':
        return ['supervised']
    raise ValueError(f"Unknown training: {training}")


def load_test_full_series(loss_dir, training):
    """Concatenate test_full across all recorded phases. Returns array or None."""
    phases = infer_phase_names(training, loss_dir)
    ys = []
    for phase in phases:
        csv_path = os.path.join(loss_dir, f'loss_{phase}.csv')
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path, index_col='epoch_in_phase')
        if 'test_full' not in df.columns:
            continue
        ys.append(df['test_full'].to_numpy())
    if not ys:
        return None
    return np.concatenate(ys)


def run_name(model_type, training, seed=None):
    tag = {'multistage': 'ms', 'multistage_nr': 'ms_nr',
           'singlestage': 'ss', 'supervised': 'sup'}[training]
    ep  = {'multistage': EPOCHS, 'multistage_nr': EPOCHS,
           'singlestage': SS_EPOCHS, 'supervised': SUP_EPOCHS}[training]
    name = (f'{model_type}_{tag}_{NTRAIN}_Adam_ep{ep}'
            f'_visc{NU}_m{MODES}_w{WIDTH}_T{T}')
    if seed is not None:
        name += f'_seed{seed}'
    return name


def loss_dir_for(model_type, training, seed):
    """Return the loss directory for a given (model, training, seed)."""
    if seed == 0:
        # Seed 0 is in the main experiment directory (no _seed suffix)
        exact = os.path.join(MAIN_LOSS_DIR, run_name(model_type, training))
        if os.path.isdir(exact):
            return exact
        # Fallback: glob for legacy naming variants
        tag = {'multistage': 'ms', 'multistage_nr': 'ms_nr',
               'singlestage': 'ss', 'supervised': 'sup'}[training]
        ep  = {'multistage': EPOCHS, 'multistage_nr': EPOCHS,
               'singlestage': SS_EPOCHS, 'supervised': SUP_EPOCHS}[training]
        pat = (f'{model_type}_{tag}_{NTRAIN}_Adam_ep*'
               f'_visc{NU}_m{MODES}_w{WIDTH}_T{T}')
        candidates = glob(os.path.join(MAIN_LOSS_DIR, pat))
        # Exclude any that have _seed in name (those belong to ablation)
        candidates = [c for c in candidates if '_seed' not in os.path.basename(c)]
        if candidates:
            return max(candidates,
                       key=lambda p: int(os.path.basename(p)
                                         .split('_Adam_ep', 1)[1]
                                         .split('_', 1)[0]))
        return exact
    else:
        # Seeds 11, 22, 33 are in the ablation directory
        path = os.path.join(ABLATION_LOSS_DIR, run_name(model_type, training, seed))
        return path


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # Header
    col_w = [10, 15, 22, 8]
    header = (f"{'Model':<{col_w[0]}}{'Training':<{col_w[1]}}"
              f"{'Mean ± Std (plateau)':<{col_w[2]}}{'Seeds':<{col_w[3]}}")
    sep    = '-' * sum(col_w)
    print(f"\n{'=' * sum(col_w)}")
    print(f"  Convergence plateau — NS nu={NU}, seeds {SEEDS}")
    print(f"{'=' * sum(col_w)}")
    print(header)
    print(sep)

    results = {}

    for model_type in MODELS:
        first_row = True
        for training in TRAININGS:
            plateaus   = []
            found_seeds = []
            missing    = []

            for seed in SEEDS:
                ld = loss_dir_for(model_type, training, seed)
                series = load_test_full_series(ld, training)
                if series is None or len(series) == 0:
                    missing.append(seed)
                    continue
                mean_p, _ = final_plateau_stats(series)
                if not np.isnan(mean_p):
                    plateaus.append(mean_p)
                    found_seeds.append(seed)
                else:
                    missing.append(seed)

            # Aggregate across seeds
            if len(plateaus) == 0:
                mean_str = 'N/A'
                seeds_str = f'0/{len(SEEDS)}'
            elif len(plateaus) == 1:
                mean_str  = f'{plateaus[0]:.4e}  (±  N/A)'
                seeds_str = f'1/{len(SEEDS)}'
            else:
                m   = np.mean(plateaus)
                sd  = np.std(plateaus, ddof=1)
                mean_str  = f'{m:.4e}  ±  {sd:.4e}'
                seeds_str = f'{len(plateaus)}/{len(SEEDS)}'

            model_label = model_type if first_row else ''
            first_row   = False

            print(f"{model_label:<{col_w[0]}}{training:<{col_w[1]}}"
                  f"{mean_str:<{col_w[2]}}{seeds_str:<{col_w[3]}}")

            results[(model_type, training)] = {
                'plateaus':    plateaus,
                'found_seeds': found_seeds,
                'missing':     missing,
            }

        print(sep)

    # Per-seed detail (useful for spotting outliers)
    print(f"\n{'=' * sum(col_w)}")
    print("  Per-seed detail")
    print(f"{'=' * sum(col_w)}")
    seed_header = (f"{'Model':<{col_w[0]}}{'Training':<{col_w[1]}}"
                   + ''.join(f'seed{s:<7}' for s in SEEDS))
    print(seed_header)
    print('-' * (sum(col_w) + 8 * len(SEEDS)))

    for model_type in MODELS:
        first_row = True
        for training in TRAININGS:
            row_vals = []
            for seed in SEEDS:
                ld     = loss_dir_for(model_type, training, seed)
                series = load_test_full_series(ld, training)
                if series is None or len(series) == 0:
                    row_vals.append('  N/A   ')
                else:
                    mean_p, _ = final_plateau_stats(series)
                    row_vals.append(f'{mean_p:.4e}' if not np.isnan(mean_p) else '  N/A   ')

            model_label = model_type if first_row else ''
            first_row   = False
            print(f"{model_label:<{col_w[0]}}{training:<{col_w[1]}}"
                  + '  '.join(f'{v:<8}' for v in row_vals))
        print('-' * (sum(col_w) + 8 * len(SEEDS)))
