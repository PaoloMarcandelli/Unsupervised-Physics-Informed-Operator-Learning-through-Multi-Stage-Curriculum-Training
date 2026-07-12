"""
train_model_ablation.py — Multi-seed ablation for NS-vorticity (nu=1e-3).

Identical to train_model.py for exp_dir=navier-stokes-vorticity, but:
  - accepts --seed  (for seeds 11, 22, 33 — seed 0 is in the main experiment)
  - supports all five models  (phisfno | pino | unet | resnet | tfnet)
  - supports all four training strategies (multistage | multistage_nr | singlestage | supervised)
  - writes loss CSVs to  ablation/loss/<run_name>/
  - writes model to      ablation/model/<run_name>
  where <run_name> appends _seed{seed} to the standard naming convention.

Usage (from the Navier_Stokes/ directory):
    python navier-stokes-vorticity/ablation/train_model_ablation.py \
        --model phisfno --training multistage --seed 11
"""

import argparse
import math
import operator
import os
import sys
from functools import reduce
from timeit import default_timer

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
# Script lives in navier-stokes-vorticity/ablation/
_HERE       = os.path.dirname(os.path.abspath(__file__))   # .../navier-stokes-vorticity/ablation
_NS_DIR     = os.path.dirname(_HERE)                        # .../navier-stokes-vorticity
_ROOT       = os.path.dirname(_NS_DIR)                      # .../Navier_Stokes
_SHARED_DIR = os.path.join(_ROOT, 'shared')

for _p in (_SHARED_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utilities3 import LpLoss, LpLossSafe, MatReader
from spline_models import interpolate_states
from setups import Dataset
from models import Net2d, fluid_model, ResNet18_NS, TFNet_NS, PINONet2d


# ════════════════════════════════════════════════════════════════════════════
# FIXED HYPERPARAMETERS  (match defaults used in run_experiments.sh for NS nu=1e-3)
# ════════════════════════════════════════════════════════════════════════════

NTRAIN      = 16
NTEST       = 2
MODES       = 12
WIDTH       = 20
BSIZE       = 2          # actual mini-batch size  (= params.bsize in train_model.py)
EPOCHS      = 100        # epochs per phase (multistage / multistage_nr)
SS_EPOCHS   = 200        # epochs for singlestage
SUP_EPOCHS  = 300        # epochs for supervised
S           = 64
T_IN        = 10
T           = 10
STEP        = 1
NU          = 1e-3
OFFSET      = 40
SUB         = 4          # spatial subsampling 256 → 64
LR          = 2e-3
SCHED_STEP  = 20
SCHED_GAMMA = 0.5
ORDERS_V    = [2, 2]
N_SAMPLES   = 4
RF          = 8
PAD         = 1
DT          = 0.25       # snapshot spacing in NS dataset

RESNET_WIDTH  = 64
TFNET_WIDTH   = 64
TFNET_DEPTH   = 4

DATA_PATH = os.path.join(_NS_DIR, f'ns_data_{NU}.mat')
OUT_BASE  = _HERE        # navier-stokes-vorticity/ablation/


# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS  (identical to train_model.py)
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


def make_border_mask(Sg, device, thickness=1):
    bm = torch.zeros(1, Sg, Sg, 1, device=device)
    t  = thickness
    bm[:, :t,  :, :] = 1.0
    bm[:, -t:, :, :] = 1.0
    bm[:, :, :t,  :] = 1.0
    bm[:, :, -t:, :] = 1.0
    return bm


# ════════════════════════════════════════════════════════════════════════════
# PHASE DEFINITIONS  (NS only — no KH / Kolmogorov branches)
# ════════════════════════════════════════════════════════════════════════════

def build_phases(training, model_type):
    lam_reg = {}
    if model_type in ('phisfno', 'unet'):
        lam_reg = {'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2}

    if training == 'multistage':
        phases = [
            {'name': 'boundary_only_1', 'record': False, 'epochs': EPOCHS,
             'λ_sup': 1.0, 'λ_res': 0.0, **lam_reg},
            {'name': 'boundary_only_2', 'record': False, 'epochs': EPOCHS,
             'λ_sup': 1.0, 'λ_res': 0.0, **lam_reg},
            {'name': 'full_loss_1',     'record': True,  'epochs': EPOCHS,
             'λ_sup': 0.8, 'λ_res': 0.5, **lam_reg},
            {'name': 'full_loss_2',     'record': True,  'epochs': EPOCHS,
             'λ_sup': 0.5, 'λ_res': 1.0, **lam_reg},
            {'name': 'full_loss_3',     'record': True,  'epochs': EPOCHS,
             'λ_sup': 0.2, 'λ_res': 1.5, **lam_reg},
        ]
        reset_optimizer = True

    elif training == 'multistage_nr':
        phases, _ = build_phases('multistage', model_type)
        reset_optimizer = False

    elif training == 'singlestage':
        phases = [
            {'name': 'full_loss_1', 'record': True, 'epochs': SS_EPOCHS,
             'λ_sup': 1.0, 'λ_res': 1.0, **lam_reg},
        ]
        reset_optimizer = False

    elif training == 'supervised':
        phases = [
            {'name': 'supervised', 'record': True,  'epochs': SUP_EPOCHS,
             'λ_sup': 1.0, 'λ_res': 0.0},
        ]
        reset_optimizer = False

    return phases, reset_optimizer


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-seed ablation — NS-vorticity nu=1e-3, all models & strategies')
    parser.add_argument('--model',    choices=['phisfno', 'pino', 'unet', 'resnet', 'tfnet'],
                        required=True)
    parser.add_argument('--training', choices=['multistage', 'multistage_nr',
                                               'singlestage', 'supervised'],
                        required=True)
    parser.add_argument('--seed',     type=int, required=True,
                        help='Random seed (e.g. 11, 22, 33)')
    args = parser.parse_args()

    model_type = args.model
    training   = args.training
    seed       = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    # ── Output paths ──────────────────────────────────────────────────────────
    training_tag = {
        'multistage':    'ms',
        'multistage_nr': 'ms_nr',
        'singlestage':   'ss',
        'supervised':    'sup',
    }[training]
    ep_label = {
        'multistage':    EPOCHS,
        'multistage_nr': EPOCHS,
        'singlestage':   SS_EPOCHS,
        'supervised':    SUP_EPOCHS,
    }[training]

    run_name      = (f'{model_type}_{training_tag}_{NTRAIN}_Adam_ep{ep_label}'
                     f'_visc{NU}_m{MODES}_w{WIDTH}_T{T}_seed{seed}')
    path_loss_dir = os.path.join(OUT_BASE, 'loss',  run_name)
    path_model    = os.path.join(OUT_BASE, 'model', run_name)
    os.makedirs(path_loss_dir, exist_ok=True)
    os.makedirs(os.path.join(OUT_BASE, 'model'), exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
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

    pino_shuffle = (model_type in ('pino', 'resnet', 'tfnet'))
    _g = torch.Generator()
    _g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u.to(device)),
        batch_size=BSIZE, shuffle=pino_shuffle,
        generator=_g if pino_shuffle else None)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u.to(device)),
        batch_size=BSIZE, shuffle=False)

    batch_size = BSIZE   # matches train_model.py: batch_size = params.bsize

    # ── Build model ───────────────────────────────────────────────────────────
    if model_type == 'phisfno':
        model = Net2d(MODES, WIDTH).to(device)
    elif model_type == 'pino':
        model = PINONet2d(MODES, WIDTH).to(device)
    elif model_type == 'unet':
        model = fluid_model(ORDERS_V, zero_mean=True, use_tanh=True).to(device)
    elif model_type == 'resnet':
        model = ResNet18_NS(T_in=T_IN, width=RESNET_WIDTH).to(device)
    elif model_type == 'tfnet':
        model = TFNet_NS(T_in=T_IN, width=TFNET_WIDTH, fusion_depth=TFNET_DEPTH).to(device)

    n_params = sum(reduce(operator.mul, p.size()) for p in model.parameters())
    print(f"[ablation] model={model_type} | training={training} | "
          f"seed={seed} | nu={NU} | params={n_params}")

    # ── Loss / setup ──────────────────────────────────────────────────────────
    loss_data = LpLossSafe(size_average=True) if model_type == 'unet' else LpLoss(size_average=True)
    loss_bd   = LpLoss(size_average=True)

    use_spline    = model_type in ('phisfno', 'unet')
    is_supervised = (training == 'supervised')

    v_size   = int(np.prod([i + 1 for i in ORDERS_V]))
    dataset  = Dataset(batch_size, v_size, S, RF, N_SAMPLES, PAD) if use_spline else None
    bm_const = make_border_mask(S, device) if not use_spline else None

    # NS forcing term
    x_grid = torch.linspace(0, 1, S + 1, device=device)[:-1]
    X, Y   = torch.meshgrid(x_grid, x_grid, indexing='ij')
    f0 = (0.1 * (torch.sin(2 * math.pi * (X + Y)) +
                 torch.cos(2 * math.pi * (X + Y)))).unsqueeze(0)  # (1, S, S)

    dealias = make_dealias_mask(S, device).unsqueeze(0)  # (1, S, S)
    n_steps = T // STEP

    # ── Phases ────────────────────────────────────────────────────────────────
    phases, reset_optimizer = build_phases(training, model_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    print(f'\n{"=" * 65}')
    print(f'  {run_name}')
    print(f'{"=" * 65}\n')

    # ══════════════════════════════════════════════════════════════════════════
    # Phase loop
    # ══════════════════════════════════════════════════════════════════════════
    for phase in phases:
        print(f"\n=== Phase '{phase['name']}' | epochs {phase['epochs']} ===\n")

        if reset_optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

        λ_sup    = phase['λ_sup']
        λ_res    = phase['λ_res']
        λ_smooth = phase.get('λ_smooth', 0.0)
        λ_tv     = phase.get('λ_tv',     0.0)
        λ_mean   = phase.get('λ_mean',   0.0)

        train_res_hist, train_bd_hist, train_full_hist = [], [], []
        test_step_hist,  test_full_hist                = [], []

        # ── Epoch loop ────────────────────────────────────────────────────────
        for ep in range(phase['epochs']):
            model.train()
            t_start = default_timer()

            ep_res_sum = ep_bd_sum = ep_step_sum = ep_full_sum = 0.0

            # ── Batch loop ────────────────────────────────────────────────────
            for xx, yy in train_loader:
                xx, yy = toCuda([xx, yy])
                optimizer.zero_grad()

                if use_spline:
                    v_mask        = toCuda(dataset.ask())
                    interior_mask = (1 - v_mask).unsqueeze(1)
                    bm_bool       = (v_mask[0] > 0).reshape(-1)
                else:
                    bd = torch.zeros(batch_size, S, S, device=device)
                    bd[:, :PAD,  :] = 1.0
                    bd[:, -PAD:, :] = 1.0
                    bd[:, :, :PAD]  = 1.0
                    bd[:, :, -PAD:] = 1.0
                    interior_mask = (1.0 - bd)

                loss_total = 0.0

                # ── Autoregressive time loop ───────────────────────────────
                for t in range(0, T, STEP):
                    y  = yy[..., t:t + STEP]
                    im = model(xx)

                    if use_spline:
                        ω, v_field, grad_ω, lap_ω = interpolate_states(
                            im,
                            offset=torch.tensor([0.0, 0.0], device=device),
                            orders_v=ORDERS_V)
                        omega_now  = ω[..., 0]
                        omega_prev = xx[..., -1]
                    else:
                        ω          = im
                        v_field    = None
                        omega_now  = im[..., 0]
                        omega_prev = xx[..., -1]

                    # Boundary / supervised loss
                    if is_supervised:
                        loss_sup_step = loss_bd(
                            ω.reshape(ω.size(0), -1),
                            y.reshape(y.size(0), -1))
                    elif use_spline:
                        loss_sup_step = lp_on_mask(loss_bd, ω, y, bm_bool)
                    else:
                        bm = bm_const.repeat(im.shape[0], 1, 1, 1)
                        loss_sup_step = loss_bd(
                            (im * bm).reshape(im.shape[0], -1),
                            (y  * bm).reshape(y.shape[0],  -1))

                    # Physics residual
                    if λ_res > 0.0:
                        dω_dt = (omega_now - omega_prev) / DT
                        _, _, _, _, lapw, conv = spectral_ops(omega_now, dealias)
                        r = dω_dt + conv - NU * lapw - f0.expand(xx.size(0), -1, -1)
                        if use_spline:
                            res = r.unsqueeze(1)
                            wgt = interior_mask
                        else:
                            res = r
                            wgt = interior_mask
                        loss_res_step = (res**2 * wgt).sum() / wgt.sum().clamp_min(1)
                    else:
                        loss_res_step = torch.tensor(0.0, device=device)

                    # Spline regularisation
                    smooth_loss = tv_loss = mean_v_loss = torch.tensor(0.0, device=device)
                    if use_spline and not is_supervised:
                        w  = omega_now
                        dx = w[:, 1:, :] - w[:, :-1, :]
                        dy = w[:, :, 1:] - w[:, :, :-1]
                        smooth_loss = dx.pow(2).mean() + dy.pow(2).mean()
                        tv_loss     = dx.abs().mean()  + dy.abs().mean()
                        if v_field is not None and λ_mean > 0.0:
                            vx = v_field[..., 0].mean(dim=(-2, -1))
                            vy = v_field[..., 1].mean(dim=(-2, -1))
                            mean_v_loss = (vx.pow(2) + vy.pow(2)).mean()

                    loss_step = (λ_sup    * loss_sup_step
                                 + λ_res    * loss_res_step
                                 + λ_smooth * smooth_loss
                                 + λ_tv     * tv_loss
                                 + λ_mean   * mean_v_loss)

                    loss_total = loss_total + loss_step

                    ep_res_sum  += loss_res_step.item()
                    ep_bd_sum   += loss_sup_step.item()
                    ep_step_sum += loss_step.item()

                    next_frame = y if is_supervised else ω
                    pred = ω if t == 0 else torch.cat([pred, ω], dim=-1)
                    xx   = torch.cat([gridx.repeat(batch_size, 1, 1, 1),
                                      gridy.repeat(batch_size, 1, 1, 1),
                                      xx[..., 2 + STEP:], next_frame], dim=-1)

                loss_total.backward()
                optimizer.step()
                ep_full_sum += loss_data(
                    pred.reshape(batch_size, -1),
                    yy.reshape(batch_size, -1)).item()

            # ── Normalise epoch logs ──────────────────────────────────────────
            nb    = len(train_loader)
            denom = nb * n_steps
            train_res_mean  = ep_res_sum  / max(1, denom)
            train_bd_mean   = ep_bd_sum   / max(1, denom)
            train_step_mean = ep_step_sum / max(1, denom)
            train_full_mean = ep_full_sum / max(1, nb)

            # ── Eval ──────────────────────────────────────────────────────────
            model.eval()
            test_step_sum = test_full_sum = 0.0
            with torch.no_grad():
                for xx, yy in test_loader:
                    xx, yy = toCuda([xx, yy])
                    pred_t = None
                    for t in range(0, T, STEP):
                        y  = yy[..., t:t + STEP]
                        im = model(xx)
                        if use_spline:
                            omega, *_ = interpolate_states(
                                im,
                                offset=torch.tensor([0.0, 0.0], device=device),
                                orders_v=ORDERS_V)
                        else:
                            omega = im

                        test_step_sum += loss_data(
                            omega.reshape(batch_size, -1),
                            y.reshape(batch_size, -1)).item()

                        pred_t = omega if pred_t is None else torch.cat([pred_t, omega], dim=-1)
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

            print(
                f"Ep {ep + 1:3d}/{phase['epochs']} │ {elapsed:6.2f}s │ "
                f"bd {train_bd_mean:8.4e} │ res {train_res_mean:8.4e} │ "
                f"step {train_step_mean:8.4e} │ full {train_full_mean:8.4e} │ "
                f"test_step {test_step_mean:8.4e} │ test_full {test_full_mean:8.4e}")

            train_bd_hist.append(train_bd_mean)
            train_full_hist.append(train_full_mean)
            test_step_hist.append(test_step_mean)
            test_full_hist.append(test_full_mean)
            if phase['record']:
                train_res_hist.append(train_res_mean)

        # ── Save phase CSV ─────────────────────────────────────────────────────
        if phase['record']:
            idx  = list(range(1, phase['epochs'] + 1))
            data = {'train_bd':   train_bd_hist,
                    'train_full': train_full_hist,
                    'test_full':  test_full_hist,
                    'test_step':  test_step_hist,
                    'train_res':  train_res_hist}
            df = pd.DataFrame(data, index=idx)
            df.index.name = 'epoch_in_phase'
            csv_path = os.path.join(path_loss_dir, f"loss_{phase['name']}.csv")
            df.to_csv(csv_path)
            print(f"→ Saved: {csv_path}")

    # ── Save final model ───────────────────────────────────────────────────────
    torch.save(model, path_model)
    print(f"→ Model saved: {path_model}")
    print('\nDone.')
