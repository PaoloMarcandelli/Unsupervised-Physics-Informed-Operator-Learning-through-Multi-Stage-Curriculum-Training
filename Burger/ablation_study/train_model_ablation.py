"""
train_model_ablation.py — PhIS-FNO ablation study on phase lambda weights.

Runs 4 lambda configurations × 2 training modes (multistage, multistage_nr)
= 8 training runs total, then generates a 2×2 comparison figure.

Usage (from the Burger/ directory):
    conda run -n phisfno python ablation_study/train_model_ablation.py

Output structure (inside ablation_study/):
    loss/<cfg_key>_<training>/loss_full_loss_{1,2,3}.csv
    model/<cfg_key>_<training>
    figures/ablation_comparison.png
"""

import os, sys, math
from timeit import default_timer

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Add parent Burger/ directory to import path ──────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))   # .../Burger/ablation_study
_PARENT = os.path.dirname(_HERE)                        # .../Burger
sys.path.insert(0, _PARENT)

from utilities3 import MatReader, LpLoss
from spline_models1d import interpolate_states
from models import Net1d

# ════════════════════════════════════════════════════════════════════════════
# FIXED HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════

SEED        = 0
NTRAIN      = 90
NTEST       = 10
SUB         = 4
NU          = 0.1
DT          = 0.1
LDOM        = 2 * math.pi
MODES       = 32
WIDTH       = 64
EPOCHS      = 100       # epochs per phase
BATCH_SIZE  = 10
LR          = 1e-2
SCHED_STEP  = 10
SCHED_GAMMA = 0.5
K_BD        = 100

# Spline regularisation — kept constant across all ablation configs
LAM_SMOOTH  = 1e-1
LAM_TV      = 1e-2
LAM_MEAN    = 1e-2

DATA_PATH   = os.path.join(_PARENT, 'dataset', 'burgers1d_dataset.mat')
OUT_BASE    = _HERE    # all outputs go inside ablation_study/

# ════════════════════════════════════════════════════════════════════════════
# LAMBDA CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════════════

LAMBDA_CONFIGS = [
    {
        'id':   1,
        'key':  'lam1',
        'name': 'Baseline',
        'phases': [
            {'λ_sup': 0.8,  'λ_res': 0.5},
            {'λ_sup': 0.5,  'λ_res': 1.0},
            {'λ_sup': 0.5,  'λ_res': 1.5},
        ],
    },
    {
        'id':   2,
        'key':  'lam2',
        'name': 'Low-lambda',
        'phases': [
            {'λ_sup': 1.0,  'λ_res': 0.0},
            {'λ_sup': 0.5,  'λ_res': 0.5},
            {'λ_sup': 0.25, 'λ_res': 0.75},
        ],
    },
    {
        'id':   3,
        'key':  'lam3',
        'name': 'High-lambda',
        'phases': [
            {'λ_sup': 1.0,  'λ_res': 0.0},
            {'λ_sup': 1.0,  'λ_res': 0.5},
            {'λ_sup': 1.0,  'λ_res': 1.0},
        ],
    },
    {
        'id':   4,
        'key':  'lam4',
        'name': 'Shape-changed',
        'phases': [
            {'λ_sup': 1.0,  'λ_res': 0.2},
            {'λ_sup': 0.7,  'λ_res': 0.8},
            {'λ_sup': 0.4,  'λ_res': 1.2},
        ],
    },
]

TRAININGS = ['multistage', 'multistage_nr']

# ════════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ════════════════════════════════════════════════════════════════════════════

def loss_dir(cfg_key, training):
    return os.path.join(OUT_BASE, 'loss', f'{cfg_key}_{training}')

def model_dir(cfg_key, training):
    return os.path.join(OUT_BASE, 'model', f'{cfg_key}_{training}')

# ════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def lp_on_mask(myloss, x, y, mask):
    if x.dim() == 3: x = x.squeeze(-1)
    if y.dim() == 3: y = y.squeeze(-1)
    return myloss(x[:, mask], y[:, mask])

def build_boundary_mask(S, k_bd, device):
    m = torch.zeros(S, dtype=torch.bool, device=device)
    k = min(k_bd, S // 2)
    m[:k] = True;  m[-k:] = True
    return m

# ════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_training(cfg, training, device,
                 x_train, y_train, x_test, y_test,
                 border_mask, interior_mask):

    reset = (training == 'multistage')
    path_loss  = loss_dir(cfg['key'], training)
    path_model = model_dir(cfg['key'], training)
    os.makedirs(path_loss, exist_ok=True)
    os.makedirs(path_model, exist_ok=True)

    # Reset seed so ms and ms_nr start from identical model weights
    # and see the same batch ordering in phase 1.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    _g = torch.Generator()
    _g.manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=_g)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False)

    model  = Net1d(MODES, WIDTH).to(device)
    myloss = LpLoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHED_STEP, SCHED_GAMMA)

    for phase_idx, phase_lam in enumerate(cfg['phases']):
        ph_name = f'full_loss_{phase_idx + 1}'
        λ_sup   = phase_lam['λ_sup']
        λ_res   = phase_lam['λ_res']

        if reset:
            # multistage: fresh Adam + scheduler at every phase
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHED_STEP, SCHED_GAMMA)
        else:
            # multistage_nr: keep Adam moments, reset only LR schedule
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHED_STEP, SCHED_GAMMA)

        bd_hist, res_hist, test_hist = [], [], []

        for ep in range(EPOCHS):
            t0 = default_timer()
            model.train()
            ep_bd = ep_res = 0.0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)

                im = model(x)

                # boundary loss at grid nodes (offset=0)
                u_bd, _, _ = interpolate_states(
                    im, offset=torch.tensor([0.0], device=device), orders_v=[2])
                loss_bd = lp_on_mask(myloss, u_bd, y, border_mask)

                # physics residual at midpoints (offset=0.5)
                if λ_res > 0.0:
                    u_r, ux, uxx = interpolate_states(
                        im, offset=torch.tensor([0.5], device=device), orders_v=[2])
                    d_u_dt   = (u_r - x[..., 0]) / DT
                    residual = d_u_dt + u_r * ux - NU * uxx
                    loss_res = (residual[:, interior_mask] ** 2).mean()
                else:
                    loss_res = torch.tensor(0.0, device=device)
                    u_r = u_bd

                # spline regularisation (same for all configs)
                ddx         = u_r[:, 1:] - u_r[:, :-1]
                smooth_loss = ddx.pow(2).mean()
                tv_loss     = ddx.abs().mean()
                mean_loss   = u_r.mean(dim=-1).pow(2).mean()

                loss = (λ_sup * loss_bd + λ_res * loss_res
                        + LAM_SMOOTH * smooth_loss
                        + LAM_TV     * tv_loss
                        + LAM_MEAN   * mean_loss)
                loss.backward()
                optimizer.step()

                ep_bd  += loss_bd.item()
                ep_res += loss_res.item()

            scheduler.step()

            # evaluation
            model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    im = model(x)
                    u_pred, _, _ = interpolate_states(
                        im, offset=torch.tensor([0.0], device=device), orders_v=[2])
                    test_l2 += myloss(
                        u_pred.reshape(u_pred.size(0), -1),
                        y.reshape(y.size(0), -1)).item()

            ep_bd   /= NTRAIN
            ep_res  /= len(train_loader)
            test_l2 /= NTEST

            bd_hist.append(ep_bd)
            res_hist.append(ep_res)
            test_hist.append(test_l2)

            print(f'  [{cfg["key"]} {training}] Ph{phase_idx+1} '
                  f'Ep{ep+1:3d}/{EPOCHS}  '
                  f'bd={ep_bd:.3e}  res={ep_res:.3e}  test={test_l2:.3e}  '
                  f'({default_timer()-t0:.1f}s)', flush=True)

        # save phase CSV
        df = pd.DataFrame(
            {'train_bd': bd_hist, 'train_res': res_hist, 'test_full': test_hist},
            index=range(1, EPOCHS + 1))
        df.index.name = 'epoch_in_phase'
        csv_path = os.path.join(path_loss, f'loss_{ph_name}.csv')
        df.to_csv(csv_path)
        print(f'  → {csv_path}')

    torch.save(model, os.path.join(path_model, 'model.pt'))
    print(f'  → Model saved: {path_model}/model.pt')

# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def load_test_series(cfg_key, training):
    """Concatenate test_full across the 3 phases. Returns (x, y) or (None, None)."""
    ld = loss_dir(cfg_key, training)
    xs, ys, cum = [], [], 0
    for i in range(1, 4):
        f = os.path.join(ld, f'loss_full_loss_{i}.csv')
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


def plot_ablation():
    fig_dir = os.path.join(OUT_BASE, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    C_MS   = '#00C853'   # green  — multistage reset
    C_MSNR = '#0062B2'   # blue   — multistage no-reset

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.flatten()

    for ax, cfg in zip(axes, LAMBDA_CONFIGS):
        i = cfg['id']

        # ms  (green)
        x_ms, y_ms = load_test_series(cfg['key'], 'multistage')
        if x_ms is not None:
            ax.plot(x_ms, smooth_log(y_ms), color=C_MS, linewidth=2.0,
                    label=f'$\\lambda_{i}$ reset')

        # ms_nr  (blue)
        x_nr, y_nr = load_test_series(cfg['key'], 'multistage_nr')
        if x_nr is not None:
            ax.plot(x_nr, smooth_log(y_nr), color=C_MSNR, linewidth=2.0,
                    label=f'$\\lambda_{i}$ no reset')

        # phase dividers
        for p in [EPOCHS, 2 * EPOCHS]:
            ax.axvline(p, color='k', linewidth=0.7, linestyle='--', alpha=0.6)

        # title with lambda values
        ph = cfg['phases']
        ph_str = '  |  '.join(
            f'Ph{k+1}: $\\lambda_{{sup}}$={ph[k]["λ_sup"]}  $\\lambda_{{res}}$={ph[k]["λ_res"]}'
            for k in range(3))
        ax.set_title(
            f'{cfg["name"]}  ($\\lambda_{i}$)\n{ph_str}',
            fontsize=9.5, fontweight='bold')

        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(r'Relative $L_2$ test error', fontsize=11)
        ax.grid(True, which='both', ls=':', alpha=0.4)
        ax.legend(fontsize=11, framealpha=0.9)

    fig.suptitle('PhIS-FNO — Lambda ablation study  (seed=0)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(fig_dir, 'ablation_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n→ Saved: {out}')


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    S  = (2 ** 13) // SUB    # 2048

    # load dataset
    reader = MatReader(DATA_PATH)
    x_data = reader.read_field('a')[:, ::SUB]
    y_data = reader.read_field('u')[:, -1, ::SUB]

    x_train = x_data[:NTRAIN];  y_train = y_data[:NTRAIN]
    x_test  = x_data[-NTEST:];  y_test  = y_data[-NTEST:]

    grid    = torch.tensor(
        np.linspace(0, LDOM, S).reshape(1, S, 1), dtype=torch.float)
    x_train = torch.cat([x_train.reshape(NTRAIN, S, 1), grid.repeat(NTRAIN, 1, 1)], dim=2)
    x_test  = torch.cat([x_test.reshape(NTEST,  S, 1), grid.repeat(NTEST,  1, 1)], dim=2)

    border_mask   = build_boundary_mask(S, K_BD, device)
    interior_mask = ~border_mask

    # ── Training loop ─────────────────────────────────────────────────────────
    for cfg in LAMBDA_CONFIGS:
        for training in TRAININGS:
            print(f'\n{"=" * 65}', flush=True)
            print(f'  Config {cfg["id"]}: {cfg["name"]} ({cfg["key"]})  |  '
                  f'Training: {training}', flush=True)
            print(f'{"=" * 65}', flush=True)
            run_training(cfg, training, device,
                         x_train, y_train, x_test, y_test,
                         border_mask, interior_mask)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_ablation()
    print('\nDone.')
