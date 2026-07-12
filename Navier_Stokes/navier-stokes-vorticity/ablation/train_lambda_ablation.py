"""
train_lambda_ablation.py — PhIS-FNO lambda-weight ablation on NS nu=1e-3.

Trains PhIS-FNO with multistage (Adam reset) and multistage_nr (no reset)
for 4 different lambda-weight configurations, then generates a 2×2 comparison
figure (one subplot per config, ms vs ms_nr curves).

Phase structure (5 phases × 100 epochs each):
  Phase 1 (boundary_only_1) — SHARED: identical for MS and MS-nr, trained once.
  Phase 2 (boundary_only_2) — MS resets optimizer; MS-nr continues same optimizer.
  Phases 3-5 (full_loss_1/2/3) — idem.

ALL phases are recorded (record=True), so the plot covers all 500 epochs.

Usage (from the Navier_Stokes/ directory):
    python navier-stokes-vorticity/ablation/train_lambda_ablation.py

Output structure (inside navier-stokes-vorticity/ablation/):
    lambda_loss/<cfg_key>_<training>/loss_<phase_name>.csv  (5 CSVs per run)
    lambda_model/<cfg_key>_<training>
    figures/lambda_ablation_comparison.png
"""

import math
import os
import sys
from timeit import default_timer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F   # noqa: F401  (kept for completeness)

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))   # .../ablation
_NS_DIR     = os.path.dirname(_HERE)                        # .../navier-stokes-vorticity
_ROOT       = os.path.dirname(_NS_DIR)                      # .../Navier_Stokes
_SHARED_DIR = os.path.join(_ROOT, 'shared')

for _p in (_SHARED_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utilities3 import LpLoss, MatReader
from spline_models import interpolate_states
from setups import Dataset
from models import Net2d


# ════════════════════════════════════════════════════════════════════════════
# FIXED HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════

SEED        = 0
NTRAIN      = 16
NTEST       = 2
MODES       = 12
WIDTH       = 20
BSIZE       = 2
EPOCHS      = 100
S           = 64
T_IN        = 10
T           = 10
STEP        = 1
NU          = 1e-3
OFFSET      = 40
SUB         = 4
LR          = 2e-3
SCHED_STEP  = 20
SCHED_GAMMA = 0.5
ORDERS_V    = [2, 2]
N_SAMPLES   = 4
RF          = 8
PAD         = 1
DT          = 0.25

# Spline regularisation — kept constant across all configs
LAM_SMOOTH  = 1e-1
LAM_TV      = 1e-2
LAM_MEAN    = 1e-2

DATA_PATH = os.path.join(_NS_DIR, f'ns_data_{NU}.mat')
OUT_BASE  = _HERE

TRAININGS = ['multistage', 'multistage_nr']

# Phase names — same for every config (5 phases × 100 epochs)
PHASE_NAMES = ['boundary_only_1', 'boundary_only_2',
               'full_loss_1', 'full_loss_2', 'full_loss_3']


# ════════════════════════════════════════════════════════════════════════════
# LAMBDA CONFIGURATIONS  (5 phases per config, ALL recorded)
# Each phase: (λ_bd, λ_res)
# ════════════════════════════════════════════════════════════════════════════

lam_reg = {'λ_smooth': LAM_SMOOTH, 'λ_tv': LAM_TV, 'λ_mean': LAM_MEAN}

LAMBDA_CONFIGS = [
    {
        'id':   1,
        'key':  'lam1',
        'name': 'Baseline',
        'phases': [
            {'name': 'boundary_only_1', 'epochs': EPOCHS,
             'λ_sup': 1.0,  'λ_res': 0.0,  **lam_reg},
            {'name': 'boundary_only_2', 'epochs': EPOCHS,
             'λ_sup': 1.0,  'λ_res': 0.0,  **lam_reg},
            {'name': 'full_loss_1',     'epochs': EPOCHS,
             'λ_sup': 0.8,  'λ_res': 0.2,  **lam_reg},
            {'name': 'full_loss_2',     'epochs': EPOCHS,
             'λ_sup': 0.5,  'λ_res': 0.7,  **lam_reg},
            {'name': 'full_loss_3',     'epochs': EPOCHS,
             'λ_sup': 0.2,  'λ_res': 1.0,  **lam_reg},
        ],
    },
    {
        'id':   2,
        'key':  'lam2',
        'name': 'Low-lambda',
        'phases': [
            {'name': 'boundary_only_1', 'epochs': EPOCHS,
             'λ_sup': 0.5,  'λ_res': 0.0,  **lam_reg},
            {'name': 'boundary_only_2', 'epochs': EPOCHS,
             'λ_sup': 0.5,  'λ_res': 0.0,  **lam_reg},
            {'name': 'full_loss_1',     'epochs': EPOCHS,
             'λ_sup': 0.4,  'λ_res': 0.1,  **lam_reg},
            {'name': 'full_loss_2',     'epochs': EPOCHS,
             'λ_sup': 0.25, 'λ_res': 0.35, **lam_reg},
            {'name': 'full_loss_3',     'epochs': EPOCHS,
             'λ_sup': 0.1,  'λ_res': 0.5,  **lam_reg},
        ],
    },
    {
        'id':   3,
        'key':  'lam3',
        'name': 'High-lambda',
        'phases': [
            {'name': 'boundary_only_1', 'epochs': EPOCHS,
             'λ_sup': 1.5,  'λ_res': 0.0,  **lam_reg},
            {'name': 'boundary_only_2', 'epochs': EPOCHS,
             'λ_sup': 1.5,  'λ_res': 0.0,  **lam_reg},
            {'name': 'full_loss_1',     'epochs': EPOCHS,
             'λ_sup': 1.2,  'λ_res': 0.3,  **lam_reg},
            {'name': 'full_loss_2',     'epochs': EPOCHS,
             'λ_sup': 0.75, 'λ_res': 1.05, **lam_reg},
            {'name': 'full_loss_3',     'epochs': EPOCHS,
             'λ_sup': 0.3,  'λ_res': 1.5,  **lam_reg},
        ],
    },
    {
        'id':   4,
        'key':  'lam4',
        'name': 'Residual-early',
        'phases': [
            {'name': 'boundary_only_1', 'epochs': EPOCHS,
             'λ_sup': 1.0,  'λ_res': 0.6,  **lam_reg},
            {'name': 'boundary_only_2', 'epochs': EPOCHS,
             'λ_sup': 1.0,  'λ_res': 0.7,  **lam_reg},
            {'name': 'full_loss_1',     'epochs': EPOCHS,
             'λ_sup': 0.8,  'λ_res': 0.9,  **lam_reg},
            {'name': 'full_loss_2',     'epochs': EPOCHS,
             'λ_sup': 0.5,  'λ_res': 1.0,  **lam_reg},
            {'name': 'full_loss_3',     'epochs': EPOCHS,
             'λ_sup': 0.2,  'λ_res': 1.2,  **lam_reg},
        ],
    },
]


# ════════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ════════════════════════════════════════════════════════════════════════════

def loss_dir(cfg_key, training):
    return os.path.join(OUT_BASE, 'lambda_loss', f'{cfg_key}_{training}')

def model_path(cfg_key, training):
    return os.path.join(OUT_BASE, 'lambda_model', f'{cfg_key}_{training}')


# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def toCuda(x):
    if isinstance(x, (tuple, list)):
        return [toCuda(xi) for xi in x]
    return x.cuda(non_blocking=True)


def spectral_ops(omega, dealias=None):
    B, Sg, _ = omega.shape
    kmax = Sg // 2
    ky = torch.cat([torch.arange(0, kmax, device=omega.device),
                    torch.arange(-kmax, 0, device=omega.device)])
    kx = ky.view(1, -1).repeat(Sg, 1).t()
    ky = ky.view(1, -1).repeat(Sg, 1)
    lap = 4 * math.pi**2 * (kx**2 + ky**2)
    lap[0, 0] = 1.0
    w_h   = torch.fft.fftn(omega, dim=(-2, -1))
    psi_h = w_h / lap
    psi_h[..., 0, 0] = 0.0
    u_hat = 1j * 2 * math.pi * ky * psi_h
    v_hat = -1j * 2 * math.pi * kx * psi_h
    u  = torch.fft.ifftn(u_hat, dim=(-2, -1)).real
    v  = torch.fft.ifftn(v_hat, dim=(-2, -1)).real
    wx = torch.fft.ifftn(1j * 2 * math.pi * kx * w_h, dim=(-2, -1)).real
    wy = torch.fft.ifftn(1j * 2 * math.pi * ky * w_h, dim=(-2, -1)).real
    lapw = torch.fft.ifftn(-lap * w_h, dim=(-2, -1)).real
    conv = u * wx + v * wy
    if dealias is not None:
        conv_h = torch.fft.fftn(conv, dim=(-2, -1))
        conv   = torch.fft.ifftn(conv_h * dealias, dim=(-2, -1)).real
    return u, v, wx, wy, lapw, conv


def make_dealias_mask(Sg, device):
    kmax = Sg // 2
    ky   = torch.cat([torch.arange(0, kmax, device=device),
                      torch.arange(-kmax, 0, device=device)])
    k_y  = ky.view(1, -1).repeat(Sg, 1)
    k_x  = k_y.t()
    return ((k_x.abs() <= (2/3)*kmax) & (k_y.abs() <= (2/3)*kmax)).float()


def lp_on_mask(myloss, x, y, mask_bool):
    X = x[..., 0].reshape(x.size(0), -1) if x.dim() == 4 else x.reshape(x.size(0), -1)
    Y = y[..., 0].reshape(y.size(0), -1) if y.dim() == 4 else y.reshape(y.size(0), -1)
    return myloss(X[:, mask_bool], Y[:, mask_bool])


# ════════════════════════════════════════════════════════════════════════════
# CORE: train one phase
# ════════════════════════════════════════════════════════════════════════════

def train_one_phase(model, optimizer, scheduler, phase,
                    save_lds,           # list of directories to save CSV (may be empty)
                    train_loader, test_loader, batch_size, n_steps,
                    gridx, gridy, dealias, f0, device):
    """
    Train `model` for one phase using the supplied optimizer/scheduler.
    Saves a CSV with all loss columns to every directory in `save_lds`.
    Returns (model, optimizer, scheduler) — same objects, updated in-place.
    """
    loss_data = LpLoss(size_average=True)
    loss_bd   = LpLoss(size_average=True)

    v_size  = int(np.prod([i + 1 for i in ORDERS_V]))
    dataset = Dataset(batch_size, v_size, S, RF, N_SAMPLES, PAD)

    λ_sup    = phase['λ_sup']
    λ_res    = phase['λ_res']
    λ_smooth = phase.get('λ_smooth', 0.0)
    λ_tv     = phase.get('λ_tv',     0.0)
    λ_mean   = phase.get('λ_mean',   0.0)

    train_res_hist, train_bd_hist, train_full_hist = [], [], []
    test_step_hist, test_full_hist                 = [], []

    for ep in range(phase['epochs']):
        model.train()
        t_start = default_timer()
        ep_res_sum = ep_bd_sum = ep_step_sum = ep_full_sum = 0.0

        for xx, yy in train_loader:
            xx, yy = toCuda([xx, yy])
            optimizer.zero_grad()

            v_mask        = toCuda(dataset.ask())
            interior_mask = (1 - v_mask).unsqueeze(1)
            bm_bool       = (v_mask[0] > 0).reshape(-1)

            loss_total = 0.0

            for t in range(0, T, STEP):
                y  = yy[..., t:t + STEP]
                im = model(xx)

                ω, v_field, grad_ω, lap_ω = interpolate_states(
                    im,
                    offset=torch.tensor([0.0, 0.0], device=device),
                    orders_v=ORDERS_V)
                omega_now  = ω[..., 0]
                omega_prev = xx[..., -1]

                loss_sup_step = lp_on_mask(loss_bd, ω, y, bm_bool)

                if λ_res > 0.0:
                    dω_dt = (omega_now - omega_prev) / DT
                    _, _, _, _, lapw, conv = spectral_ops(omega_now, dealias)
                    r = dω_dt + conv - NU * lapw - f0.expand(xx.size(0), -1, -1)
                    loss_res_step = ((r.unsqueeze(1))**2 * interior_mask).sum() \
                                    / interior_mask.sum().clamp_min(1)
                else:
                    loss_res_step = torch.tensor(0.0, device=device)

                w  = omega_now
                dx = w[:, 1:, :] - w[:, :-1, :]
                dy = w[:, :, 1:] - w[:, :, :-1]
                smooth_loss = dx.pow(2).mean() + dy.pow(2).mean()
                tv_loss     = dx.abs().mean()  + dy.abs().mean()
                if v_field is not None and λ_mean > 0.0:
                    vx = v_field[..., 0].mean(dim=(-2, -1))
                    vy = v_field[..., 1].mean(dim=(-2, -1))
                    mean_v_loss = (vx.pow(2) + vy.pow(2)).mean()
                else:
                    mean_v_loss = torch.tensor(0.0, device=device)

                loss_step = (λ_sup    * loss_sup_step
                             + λ_res    * loss_res_step
                             + λ_smooth * smooth_loss
                             + λ_tv     * tv_loss
                             + λ_mean   * mean_v_loss)

                loss_total = loss_total + loss_step
                ep_res_sum  += loss_res_step.item()
                ep_bd_sum   += loss_sup_step.item()
                ep_step_sum += loss_step.item()

                pred = ω if t == 0 else torch.cat([pred, ω], dim=-1)
                xx   = torch.cat([gridx.repeat(batch_size, 1, 1, 1),
                                  gridy.repeat(batch_size, 1, 1, 1),
                                  xx[..., 2 + STEP:], ω], dim=-1)

            loss_total.backward()
            optimizer.step()
            ep_full_sum += loss_data(
                pred.reshape(batch_size, -1),
                yy.reshape(batch_size, -1)).item()

        nb    = len(train_loader)
        denom = nb * n_steps
        train_res_mean  = ep_res_sum  / max(1, denom)
        train_bd_mean   = ep_bd_sum   / max(1, denom)
        train_full_mean = ep_full_sum / max(1, nb)

        model.eval()
        test_step_sum = test_full_sum = 0.0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx, yy = toCuda([xx, yy])
                pred_t = None
                for t in range(0, T, STEP):
                    y  = yy[..., t:t + STEP]
                    im = model(xx)
                    omega, *_ = interpolate_states(
                        im,
                        offset=torch.tensor([0.0, 0.0], device=device),
                        orders_v=ORDERS_V)
                    test_step_sum += loss_data(
                        omega.reshape(batch_size, -1),
                        y.reshape(batch_size, -1)).item()
                    pred_t = omega if pred_t is None \
                             else torch.cat([pred_t, omega], dim=-1)
                    xx = torch.cat([gridx.repeat(batch_size, 1, 1, 1),
                                    gridy.repeat(batch_size, 1, 1, 1),
                                    xx[..., 2 + STEP:], omega], dim=-1)
                test_full_sum += loss_data(
                    pred_t.reshape(batch_size, -1),
                    yy.reshape(batch_size, -1)).item()

        test_step_mean = test_step_sum / max(1, len(test_loader) * n_steps)
        test_full_mean = test_full_sum / max(1, len(test_loader))

        elapsed = default_timer() - t_start
        scheduler.step()

        print(f"  Ep {ep+1:3d}/{phase['epochs']} │ {elapsed:5.1f}s │ "
              f"bd {train_bd_mean:.3e} │ res {train_res_mean:.3e} │ "
              f"full {train_full_mean:.3e} │ test {test_full_mean:.3e}")

        train_bd_hist.append(train_bd_mean)
        train_full_hist.append(train_full_mean)
        train_res_hist.append(train_res_mean)
        test_step_hist.append(test_step_mean)
        test_full_hist.append(test_full_mean)

    # Save CSV to every requested directory
    for ld in save_lds:
        os.makedirs(ld, exist_ok=True)
        idx  = list(range(1, phase['epochs'] + 1))
        data = {'train_bd':   train_bd_hist,
                'train_full': train_full_hist,
                'test_full':  test_full_hist,
                'test_step':  test_step_hist,
                'train_res':  train_res_hist}
        df = pd.DataFrame(data, index=idx)
        df.index.name = 'epoch_in_phase'
        csv_path = os.path.join(ld, f"loss_{phase['name']}.csv")
        df.to_csv(csv_path)
        print(f"  → {csv_path}")

    return model, optimizer, scheduler


# ════════════════════════════════════════════════════════════════════════════
# TRAINING ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def run_all_for_config(cfg, device,
                       train_a, train_u, test_a, test_u,
                       gridx, gridy, dealias, f0,
                       warmup_only=False):
    """
    For a given lambda config, train:
      1. Phase 1 ONCE (identical for MS and MS-nr) — saved to BOTH loss dirs.
      2. Phases 2-5 for multistage (optimizer reset at each phase start).
      3. Phases 2-5 for multistage_nr (same optimizer continued from phase 1).

    If warmup_only=True, only phases 1-2 are trained (phases 3-5 CSVs already
    exist from a previous run and are left untouched).
    """
    ld_ms = loss_dir(cfg['key'], 'multistage')
    ld_nr = loss_dir(cfg['key'], 'multistage_nr')
    mp_ms = model_path(cfg['key'], 'multistage')
    mp_nr = model_path(cfg['key'], 'multistage_nr')
    os.makedirs(ld_ms, exist_ok=True)
    os.makedirs(ld_nr, exist_ok=True)
    os.makedirs(os.path.join(OUT_BASE, 'lambda_model'), exist_ok=True)

    # Reproducible initialisation
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    batch_size = BSIZE
    n_steps    = T // STEP

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u.to(device)),
        batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u.to(device)),
        batch_size=batch_size, shuffle=False)

    # ── Phase 1: train once, shared ───────────────────────────────────────────
    model     = Net2d(MODES, WIDTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    phase1 = cfg['phases'][0]
    print(f"\n  [{cfg['key']} SHARED] Phase '{phase1['name']}' "
          f"λ_sup={phase1['λ_sup']} λ_res={phase1['λ_res']} "
          f"(identical for MS and MS-nr)\n")

    # Save CSV to BOTH directories so load_test_series finds it in each
    model, optimizer, scheduler = train_one_phase(
        model, optimizer, scheduler, phase1,
        save_lds=[ld_ms, ld_nr],
        train_loader=train_loader, test_loader=test_loader,
        batch_size=batch_size, n_steps=n_steps,
        gridx=gridx, gridy=gridy, dealias=dealias, f0=f0, device=device)

    # Snapshot weights after phase 1 (needed to initialise MS branch)
    state_after_p1 = {k: v.clone() for k, v in model.state_dict().items()}
    # Keep (model, optimizer, scheduler) alive for the MS-nr branch below.

    # Remaining phases: 2 only (warmup_only) or 2-5 (full run)
    remaining = cfg['phases'][1:2] if warmup_only else cfg['phases'][1:]
    if warmup_only:
        print(f"\n  [warmup-only] Training phase 2 only; "
              f"phases 3-5 CSVs are kept from previous run.")

    # ── MS-nr: NO optimizer reset ─────────────────────────────────────────────
    label = "phase 2" if warmup_only else "phases 2-5"
    print(f"\n{'─'*55}")
    print(f"  [{cfg['key']} multistage_nr] {label} (continuing optimizer)")
    print(f"{'─'*55}")
    model_nr   = model      # same object, already at phase1 weights
    opt_nr     = optimizer  # same optimizer, momentum accumulated in phase 1
    sched_nr   = scheduler

    for phase in remaining:
        print(f"\n  [{cfg['key']} ms_nr] Phase '{phase['name']}' "
              f"λ_sup={phase['λ_sup']} λ_res={phase['λ_res']}\n")
        model_nr, opt_nr, sched_nr = train_one_phase(
            model_nr, opt_nr, sched_nr, phase,
            save_lds=[ld_nr],
            train_loader=train_loader, test_loader=test_loader,
            batch_size=batch_size, n_steps=n_steps,
            gridx=gridx, gridy=gridy, dealias=dealias, f0=f0, device=device)

    if not warmup_only:
        torch.save(model_nr, mp_nr)
        print(f"  → Model saved: {mp_nr}")

    # ── MS: optimizer reset at each phase ─────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  [{cfg['key']} multistage] {label} (optimizer reset each phase)")
    print(f"{'─'*55}")
    model_ms = Net2d(MODES, WIDTH).to(device)
    model_ms.load_state_dict(state_after_p1)

    for phase in remaining:
        print(f"\n  [{cfg['key']} ms] Phase '{phase['name']}' "
              f"λ_sup={phase['λ_sup']} λ_res={phase['λ_res']}\n")
        opt_ms   = torch.optim.Adam(model_ms.parameters(), lr=LR, weight_decay=1e-4)
        sched_ms = torch.optim.lr_scheduler.StepLR(
            opt_ms, step_size=SCHED_STEP, gamma=SCHED_GAMMA)
        model_ms, opt_ms, sched_ms = train_one_phase(
            model_ms, opt_ms, sched_ms, phase,
            save_lds=[ld_ms],
            train_loader=train_loader, test_loader=test_loader,
            batch_size=batch_size, n_steps=n_steps,
            gridx=gridx, gridy=gridy, dealias=dealias, f0=f0, device=device)

    if not warmup_only:
        torch.save(model_ms, mp_ms)
        print(f"  → Model saved: {mp_ms}")


# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def load_test_series(cfg_key, training):
    """Concatenate test_full across all 5 phases (boundary_only_1/2 + full_loss_1/2/3)."""
    ld = loss_dir(cfg_key, training)
    xs, ys, cum = [], [], 0
    for pname in PHASE_NAMES:
        f = os.path.join(ld, f'loss_{pname}.csv')
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


def plot_lambda_ablation():
    fig_dir = os.path.join(OUT_BASE, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    C_MS   = '#00C853'   # green  — multistage reset
    C_MSNR = '#0062B2'   # blue   — multistage no-reset

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.subplots_adjust(hspace=0.45, wspace=0.12)

    # Dividers between the 5 phases
    phase_edges = [EPOCHS * k for k in range(1, 5)]   # 100, 200, 300, 400

    for ax, cfg in zip(axes.flatten(), LAMBDA_CONFIGS):
        i = cfg['id']

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

        # Phase labels inside the axes (top), blended transform: x=data, y=axes
        blended = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        phase_centres = [EPOCHS * k + EPOCHS // 2 for k in range(5)]
        phase_labels  = ['Ph1', 'Ph2', 'Ph3', 'Ph4', 'Ph5']
        for cx, label in zip(phase_centres, phase_labels):
            ax.text(cx, 0.97, label, ha='center', va='top',
                    fontsize=7.5, color='#444444', transform=blended)

        ax.set_title(f'{cfg["name"]}', fontsize=10, fontweight='bold', pad=6)

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


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup-only', action='store_true',
                        help='Train only phases 1-2 (boundary_only). '
                             'Phases 3-5 CSVs from a previous run are kept.')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip all training and only regenerate the plot.')
    parser.add_argument('--config', type=int, default=None,
                        choices=[1, 2, 3, 4],
                        help='Run only this lambda config (1-4). Default: all.')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    reader  = MatReader(DATA_PATH)
    train_a = reader.read_field('u')[:NTRAIN,  ::SUB, ::SUB, OFFSET:OFFSET + T_IN]
    train_u = reader.read_field('u')[:NTRAIN,  ::SUB, ::SUB, OFFSET + T_IN:OFFSET + T + T_IN]
    test_a  = reader.read_field('u')[-NTEST:,  ::SUB, ::SUB, OFFSET:OFFSET + T_IN]
    test_u  = reader.read_field('u')[-NTEST:,  ::SUB, ::SUB, OFFSET + T_IN:OFFSET + T + T_IN]

    train_a = train_a.reshape(NTRAIN, S, S, T_IN)
    test_a  = test_a.reshape(NTEST,  S, S, T_IN)

    gridx = torch.linspace(0, 1, S, device=device).view(1, S, 1, 1).repeat(1, 1, S, 1)
    gridy = torch.linspace(0, 1, S, device=device).view(1, 1, S, 1).repeat(1, S, 1, 1)

    train_a = torch.cat([gridx.repeat(NTRAIN, 1, 1, 1),
                         gridy.repeat(NTRAIN, 1, 1, 1),
                         train_a.to(device)], dim=-1)
    test_a  = torch.cat([gridx.repeat(NTEST,  1, 1, 1),
                         gridy.repeat(NTEST,  1, 1, 1),
                         test_a.to(device)], dim=-1)

    # NS forcing term
    x_grid = torch.linspace(0, 1, S + 1, device=device)[:-1]
    X, Y   = torch.meshgrid(x_grid, x_grid, indexing='ij')
    f0 = (0.1 * (torch.sin(2 * math.pi * (X + Y)) +
                 torch.cos(2 * math.pi * (X + Y)))).unsqueeze(0)

    dealias = make_dealias_mask(S, device).unsqueeze(0)

    # Select configs to run
    configs_to_run = [c for c in LAMBDA_CONFIGS
                      if args.config is None or c['id'] == args.config]

    # # ── Training loop ─────────────────────────────────────────────────────────
    # if not args.plot_only:
    #     for cfg in configs_to_run:
    #         print(f'\n{"=" * 65}')
    #         print(f'  Config {cfg["id"]}: {cfg["name"]} ({cfg["key"]})'
    #               + ('  [warmup only]' if args.warmup_only else ''))
    #         print(f'{"=" * 65}')
    #         run_all_for_config(cfg, device,
    #                            train_a, train_u, test_a, test_u,
    #                            gridx, gridy, dealias, f0,
    #                            warmup_only=args.warmup_only)
    # else:
    #     print('\n[--plot-only] Skipping training, generating plot from existing CSVs.')

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_lambda_ablation()
    print('\nDone.')
