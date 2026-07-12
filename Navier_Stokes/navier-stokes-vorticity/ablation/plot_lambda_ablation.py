"""
plot_lambda_ablation.py — standalone script to regenerate lambda_ablation_comparison.png
from the already-saved loss CSVs in lambda_loss/.

Usage (from the Navier_Stokes/ directory):
    python navier-stokes-vorticity/ablation/plot_lambda_ablation.py

Or directly from the ablation/ directory:
    python plot_lambda_ablation.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../ablation

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS (must match the values used during training)
# ════════════════════════════════════════════════════════════════════════════

EPOCHS = 100
NU     = 1e-3
SEED   = 0

PHASE_NAMES = ['boundary_only_1', 'boundary_only_2',
               'full_loss_1', 'full_loss_2', 'full_loss_3']

LAMBDA_CONFIGS = [
    {'id': 1, 'key': 'lam1', 'name': 'Baseline'},
    {'id': 2, 'key': 'lam2', 'name': 'Low-lambda'},
    {'id': 3, 'key': 'lam3', 'name': 'High-lambda'},
    {'id': 4, 'key': 'lam4', 'name': 'Shape-changed'},
]

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def loss_dir(cfg_key, training):
    return os.path.join(_HERE, 'lambda_loss', f'{cfg_key}_{training}')


def load_test_series(cfg_key, training):
    """Concatenate test_full across all 5 phases."""
    ld = loss_dir(cfg_key, training)
    xs, ys, cum = [], [], 0
    for pname in PHASE_NAMES:
        f = os.path.join(ld, f'loss_{pname}.csv')
        if not os.path.isfile(f):
            print(f'  [warn] missing: {f}')
            continue
        df = pd.read_csv(f, index_col='epoch_in_phase')
        if 'test_full' not in df.columns:
            print(f'  [warn] no test_full column in: {f}')
            continue
        y = df['test_full'].to_numpy()
        xs.append(np.arange(1, len(y) + 1) + cum)
        ys.append(y)
        cum += len(y)
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def smooth_log(y, win=15, eps=1e-12):
    y = np.maximum(np.asarray(y, dtype=float), eps)
    if len(y) < win:
        win = max(3, len(y) | 1)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    yp  = np.pad(np.log10(y), (pad, pad), mode='edge')
    k   = np.ones(win) / win
    return 10 ** np.convolve(yp, k, mode='valid')


# ════════════════════════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_lambda_ablation():
    fig_dir = os.path.join(_HERE, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    C_MS   = '#00C853'   # green  — multistage reset
    C_MSNR = '#0062B2'   # blue   — multistage no-reset

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.subplots_adjust(hspace=0.45, wspace=0.12)

    # Vertical dividers between the 5 phases
    phase_edges = [EPOCHS * k for k in range(1, 5)]   # 100, 200, 300, 400

    for ax, cfg in zip(axes.flatten(), LAMBDA_CONFIGS):
        x_ms, y_ms = load_test_series(cfg['key'], 'multistage')
        if x_ms is not None:
            ax.plot(x_ms, smooth_log(y_ms), color=C_MS, linewidth=2.0,
                    label='MS (reset)')

        x_nr, y_nr = load_test_series(cfg['key'], 'multistage_nr')
        if x_nr is not None:
            ax.plot(x_nr, smooth_log(y_nr), color=C_MSNR, linewidth=2.0,
                    label='MS (no reset)')

        # Vertical dividers
        for p in phase_edges:
            ax.axvline(p, color='k', linewidth=0.7, linestyle='--', alpha=0.5)

        # Phase labels inside the axes (blended transform: x=data, y=axes)
        blended = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        phase_centres = [EPOCHS * k + EPOCHS // 2 for k in range(5)]
        phase_labels  = ['Ph1', 'Ph2', 'Ph3', 'Ph4', 'Ph5']
        for cx, label in zip(phase_centres, phase_labels):
            ax.text(cx, 0.97, label, ha='center', va='top',
                    fontsize=7.5, color='#444444', transform=blended)

        ax.set_title(cfg['name'], fontsize=10, fontweight='bold', pad=6)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch (all phases)', fontsize=10)
        ax.set_ylabel(r'Relative $L_2$ test error', fontsize=10)
        ax.set_xlim(0, 5 * EPOCHS)
        ax.grid(True, which='both', ls=':', alpha=0.4)
        ax.legend(fontsize=9, framealpha=0.9, loc='lower left')

    fig.suptitle(
        f'PhIS-FNO — $\\lambda$ ablation study  (NS $\\nu={NU}$, seed={SEED})',
        fontsize=13, fontweight='bold', y=1.01)

    out = os.path.join(fig_dir, 'lambda_ablation_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n→ Saved: {out}')


if __name__ == '__main__':
    plot_lambda_ablation()
