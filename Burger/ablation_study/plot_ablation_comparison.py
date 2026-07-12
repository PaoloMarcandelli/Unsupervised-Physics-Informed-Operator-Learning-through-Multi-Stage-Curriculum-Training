"""
plot_ablation_comparison.py — standalone plot script for the ablation study.

Reads existing loss CSVs from ablation_study/loss/ and saves
ablation_study/figures/ablation_comparison.png.

Usage (from the Burger/ directory):
    python ablation_study/plot_ablation_comparison.py

Or directly from ablation_study/:
    python plot_ablation_comparison.py
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
LOSS_DIR = os.path.join(_HERE, 'loss')
FIG_DIR  = os.path.join(_HERE, 'figures')
EPOCHS   = 100   # epochs per phase (used for vertical dividers)

# ── Lambda configurations (same structure as training script) ─────────────────
LAMBDA_CONFIGS = [
    {'key': 'lam1', 'title': 'Baseline'},
    {'key': 'lam2', 'title': 'Loss-lambda'},
    {'key': 'lam3', 'title': 'High lambda'},
    {'key': 'lam4', 'title': 'shape changed'},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_series(cfg_key, training):
    """Concatenate test_full across the 3 phases. Returns (x, y) or (None, None)."""
    run_dir = os.path.join(LOSS_DIR, f'{cfg_key}_{training}')
    xs, ys, cum = [], [], 0
    for i in range(1, 4):
        f = os.path.join(run_dir, f'loss_full_loss_{i}.csv')
        if not os.path.isfile(f):
            continue
        df = pd.read_csv(f, index_col='epoch_in_phase')
        if 'test_full' not in df.columns:
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


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_ablation():
    os.makedirs(FIG_DIR, exist_ok=True)

    C_MS   = '#00C853'   # green  — multistage (reset)
    C_MSNR = '#0062B2'   # blue   — multistage no-reset

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, cfg in zip(axes, LAMBDA_CONFIGS):
        # ── curves ──────────────────────────────────────────────────────────
        x_ms, y_ms = load_test_series(cfg['key'], 'multistage')
        if x_ms is not None:
            ax.plot(x_ms, smooth_log(y_ms), color=C_MS, linewidth=2.0,
                    label='Multistage (reset)')

        x_nr, y_nr = load_test_series(cfg['key'], 'multistage_nr')
        if x_nr is not None:
            ax.plot(x_nr, smooth_log(y_nr), color=C_MSNR, linewidth=2.0,
                    label='Multistage (no reset)')

        # ── phase dividers ───────────────────────────────────────────────────
        for p in [EPOCHS, 2 * EPOCHS]:
            ax.axvline(p, color='k', linewidth=0.8, linestyle='--', alpha=0.55)

        # ── title ─────────────────────────────────────────────────────────────
        ax.set_title(cfg['title'], fontsize=13, fontweight='bold')

        # ── axes formatting ──────────────────────────────────────────────────
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(r'Relative $L_2$ test error', fontsize=11)
        ax.grid(True, which='both', ls=':', alpha=0.4)
        ax.legend(fontsize=10, framealpha=0.9)

    fig.suptitle('PhIS-FNO — Lambda ablation study  (seed=0)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    out = os.path.join(FIG_DIR, 'ablation_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    plot_ablation()
