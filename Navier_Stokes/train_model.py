"""
train_model.py — Unified training script for NS-vorticity and Kolmogorov experiments.

Usage examples
--------------
# PhIS-FNO multistage with Adam reset, Navier-Stokes nu=1e-3
python train_model.py --exp_dir navier-stokes-vorticity \
    --model phisfno --training multistage --nu 1e-3 --offset 40

# PINO singlestage, Kolmogorov
python train_model.py --exp_dir kolmogorov \
    --model pino --training singlestage --offset 30

# UNet supervised, Navier-Stokes nu=1e-4
python train_model.py --exp_dir navier-stokes-vorticity \
    --model unet --training supervised --nu 1e-4 --offset 20

Arguments
---------
--exp_dir   : experiment directory (navier-stokes-vorticity | kolmogorov)
--model     : model architecture    (phisfno | pino | unet | resnet | tfnet)
--training  : training strategy     (multistage | multistage_nr | singlestage | supervised)
"""

import argparse
import os
import sys

# ── Parse --exp_dir before any local import so we can set sys.path first ──────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--exp_dir',
                  choices=['navier-stokes-vorticity', 'kolmogorov', 'kelvin-helmholtz'],
                  required=True)
_pre_args, _ = _pre.parse_known_args()

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_exp_dir     = os.path.join(_script_dir, _pre_args.exp_dir)
_shared_dir  = os.path.join(_script_dir, 'shared')
os.makedirs(_exp_dir, exist_ok=True)
os.chdir(_exp_dir)
sys.path.insert(0, _shared_dir)

# ── Standard imports ───────────────────────────────────────────────────────────
import math
import operator
from functools import partial, reduce
from timeit import default_timer

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities3 import LpLoss, LpLossSafe, MatReader
from spline_models import interpolate_states
from setups import Dataset
from models import Net2d, fluid_model, ResNet18_NS, TFNet_NS, PINONet2d


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def toCuda(x):
    if isinstance(x, (tuple, list)):
        return [toCuda(xi) for xi in x]
    return x.cuda(non_blocking=True)


def spectral_ops(omega, dealias=None):
    """Compute velocity (u,v), vorticity gradients (wx,wy), Laplacian lapw,
    and the de-aliased convective term conv = u·∇ω."""
    B, S, _ = omega.shape
    kmax = S // 2
    ky = torch.cat([torch.arange(0, kmax, device=omega.device),
                    torch.arange(-kmax, 0, device=omega.device)])
    kx = ky.view(1, -1).repeat(S, 1).t()
    ky = ky.view(1, -1).repeat(S, 1)

    lap = 4 * math.pi**2 * (kx**2 + ky**2)
    lap[0, 0] = 1.0  # avoid division by zero at (0,0)

    w_h   = torch.fft.fftn(omega, dim=(-2, -1))
    psi_h = w_h / lap
    psi_h[..., 0, 0] = 0.0           # zero-mean streamfunction

    u_hat = 1j * 2 * math.pi * ky * psi_h   # u = ∂ψ/∂y
    v_hat = -1j * 2 * math.pi * kx * psi_h  # v = -∂ψ/∂x
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


def spectral_div(fx, fy):
    """Spectral divergence ∂fx/∂x + ∂fy/∂y for a batch of 2-D fields.

    fx, fy : (B, S, S)  — vector field components
    Returns  (B, S, S)

    NOTE: assumes a [0,1]^2 domain with effective periodicity at the boundary.
    For KH (transmissive BCs) the interior estimate is accurate; boundary cells
    are excluded from the residual via interior_mask.
    """
    B, S, _ = fx.shape
    kmax = S // 2
    k  = torch.cat([torch.arange(0, kmax, device=fx.device),
                    torch.arange(-kmax, 0, device=fx.device)]).float()
    kx = k.view(-1, 1).expand(S, S)
    ky = k.view(1, -1).expand(S, S)
    fx_h = torch.fft.fftn(fx, dim=(-2, -1))
    fy_h = torch.fft.fftn(fy, dim=(-2, -1))
    div  = torch.fft.ifftn(
        1j * 2 * math.pi * (kx * fx_h + ky * fy_h), dim=(-2, -1)).real
    return div


def make_dealias_mask(S, device):
    """2/3-rule de-aliasing mask for a grid of size S×S."""
    kmax = S // 2
    ky   = torch.cat([torch.arange(0, kmax, device=device),
                      torch.arange(-kmax, 0, device=device)])
    k_y  = ky.view(1, -1).repeat(S, 1)
    k_x  = k_y.t()
    return ((k_x.abs() <= (2/3)*kmax) & (k_y.abs() <= (2/3)*kmax)).float()


def lp_on_mask(myloss, x, y, mask_bool):
    """Lp loss restricted to the pixels selected by mask_bool (boundary pixels).

    x, y       : (B,S,S,1) or (B,S,S)
    mask_bool  : 1-D bool tensor of length S*S
    """
    X = x[..., 0].reshape(x.size(0), -1) if x.dim() == 4 else x.reshape(x.size(0), -1)
    Y = y[..., 0].reshape(y.size(0), -1) if y.dim() == 4 else y.reshape(y.size(0), -1)
    return myloss(X[:, mask_bool], Y[:, mask_bool])


def make_border_mask(S, device, thickness=1):
    """Fixed border mask of shape (1,S,S,1), used by the PINO boundary loss."""
    bm = torch.zeros(1, S, S, 1, device=device)
    t  = thickness
    bm[:, :t,  :, :] = 1.0
    bm[:, -t:, :, :] = 1.0
    bm[:, :, :t,  :] = 1.0
    bm[:, :, -t:, :] = 1.0
    return bm



# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def get_params():
    p = argparse.ArgumentParser(
        description="Unified Navier-Stokes / Kolmogorov training script")

    # Experiment & strategy selection
    p.add_argument('--exp_dir',
                   choices=['navier-stokes-vorticity', 'kolmogorov', 'kelvin-helmholtz'],
                   required=True, help='Experiment sub-directory')
    p.add_argument('--model',    choices=['phisfno', 'pino', 'unet', 'resnet', 'tfnet'],
                   required=True, help='Model architecture')
    p.add_argument('--resnet_width', type=int, default=64,
                   help='Channel width for the ResNet-18 baseline')
    p.add_argument('--tfnet_width', type=int, default=64,
                   help='Channel width for the TF-Net baseline')
    p.add_argument('--tfnet_depth', type=int, default=4,
                   help='Number of fusion blocks in the TF-Net baseline')
    p.add_argument('--training', choices=['multistage', 'multistage_nr',
                                          'singlestage', 'supervised'],
                   required=True, help='Training strategy')

    # Dataset / model size
    p.add_argument('--ntrain',  type=int,   default=16,   help='Training samples')
    p.add_argument('--ntest',   type=int,   default=2,    help='Test samples')
    p.add_argument('--modes',   type=int,   default=12,   help='Fourier modes')
    p.add_argument('--width',   type=int,   default=20,   help='Model width')
    p.add_argument('--bsize',   type=int,   default=2,    help='Batch divisor (batch = ntrain/bsize)')

    # Training schedule
    p.add_argument('--epochs',      type=int,   default=100,
                   help='Epochs per phase (multistage / multistage_nr)')
    p.add_argument('--ss_epochs',   type=int,   default=150,
                   help='Epochs for singlestage')
    p.add_argument('--sup_epochs',  type=int,   default=150,
                   help='Epochs for supervised')
    p.add_argument('--learning_rate',    type=float, default=2e-3)
    p.add_argument('--scheduler_step',  type=int,   default=20)
    p.add_argument('--scheduler_gamma', type=float, default=0.5)

    # Grid / physics
    p.add_argument('--sub',      type=int,   default=4,    help='Spatial subsampling factor')
    p.add_argument('--S',        type=int,   default=64,   help='Spatial resolution')
    p.add_argument('--T_in',     type=int,   default=10,   help='Input sequence length')
    p.add_argument('--T',        type=int,   default=10,   help='Prediction horizon')
    p.add_argument('--step',     type=int,   default=1,    help='Autoregressive step size')
    p.add_argument('--offset',   type=int,   default=40,
                   help='Temporal offset (40=NS nu=1e-3, 20=NS nu=1e-4, 30=Kolmogorov)')
    p.add_argument('--nu',       type=float, default=1e-3, help='Kinematic viscosity')
    p.add_argument('--seed',     type=int,   default=0,    help='Random seed')
    p.add_argument('--rf',       type=int,   default=8,    help='Super-resolution factor')
    p.add_argument('--orders_v', type=int,   nargs=2, default=[2, 2],
                   help='Hermite spline orders (2D)')
    p.add_argument('--n_samples', type=int,  default=4,    help='Random field samples per batch')

    args = p.parse_args()
    args.batch_size = args.ntrain // args.bsize
    return args


# ══════════════════════════════════════════════════════════════════════════════
# Phase definitions
# ══════════════════════════════════════════════════════════════════════════════

def build_phases(training, exp_dir, epochs, ss_epochs, sup_epochs, model_type):
    """Return (phases, reset_optimizer).

    Phases carry λ_sup, λ_res and — for spline models — λ_smooth, λ_tv, λ_mean.
    reset_optimizer=True  → new Adam is created at the start of each phase (MS).
    reset_optimizer=False → a single Adam instance runs through all phases (MS-nr / SS / Sup).
    """
    lam_reg = {}
    if model_type in ('phisfno', 'unet'):   # spline-based models only
        lam_reg = {'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2}

    # Only Kolmogorov uses the 3-phase structure; NS and KH use the 5-phase structure
    is_short = (exp_dir == 'kolmogorov')

    if training == 'multistage':
        if exp_dir == 'kelvin-helmholtz':
            # KH phase schedule differs by model type:
            #   spline (phisfno/unet): stabilise splines first → λ_res=0 in phase 1.
            #   others (pino/resnet/tfnet): residual is free from step 1 → start at λ_res=0.2.
            is_spline = model_type in ('phisfno', 'unet')
            res_phase1 = 0.0 if is_spline else 0.2
            phases = [
                {'name': 'boundary_only_1', 'record': False, 'epochs': epochs,
                 'λ_sup': 1.0, 'λ_res': res_phase1, **lam_reg},
                {'name': 'full_loss_1',     'record': True,  'epochs': epochs,
                 'λ_sup': 0.5, 'λ_res': 0.5,         **lam_reg},
                {'name': 'full_loss_2',     'record': True,  'epochs': epochs,
                 'λ_sup': 0.2, 'λ_res': 1.0,         **lam_reg},
            ]
        elif is_short:  # Kolmogorov
            phases = [
                {'name': 'boundary_only_1', 'record': False, 'epochs': epochs,
                 'λ_sup': 0.5,  'λ_res': 0.1,  **lam_reg},
                {'name': 'full_loss_1',     'record': True,  'epochs': epochs,
                 'λ_sup': 0.4,  'λ_res': 0.25, **lam_reg},
                {'name': 'full_loss_2',     'record': True,  'epochs': epochs,
                 'λ_sup': 0.25, 'λ_res': 0.5,  **lam_reg},
            ]
        else:  # Navier-Stokes
            phases = [
                {'name': 'boundary_only_1', 'record': False, 'epochs': epochs,
                 'λ_sup': 1.0, 'λ_res': 0.0, **lam_reg},
                {'name': 'boundary_only_2', 'record': False, 'epochs': epochs,
                 'λ_sup': 1.0, 'λ_res': 0.0, **lam_reg},
                {'name': 'full_loss_1',     'record': True,  'epochs': epochs,
                 'λ_sup': 0.8, 'λ_res': 0.5, **lam_reg},
                {'name': 'full_loss_2',     'record': True,  'epochs': epochs,
                 'λ_sup': 0.5, 'λ_res': 1.0, **lam_reg},
                {'name': 'full_loss_3',     'record': True,  'epochs': epochs,
                 'λ_sup': 0.2, 'λ_res': 1.5, **lam_reg},
            ]
        reset_optimizer = True

    elif training == 'multistage_nr':
        # Same weights as multistage but no Adam reset between phases
        phases, _ = build_phases('multistage', exp_dir, epochs, ss_epochs, sup_epochs, model_type)
        reset_optimizer = False

    elif training == 'singlestage':
        phases = [
            {'name': 'full_loss_1', 'record': True, 'epochs': ss_epochs,
             'λ_sup': 1.0, 'λ_res': 1.0, **lam_reg},
        ]
        reset_optimizer = False

    elif training == 'supervised':
        phases = [
            {'name': 'supervised', 'record': False, 'epochs': sup_epochs,
             'λ_sup': 1.0, 'λ_res': 0.0},
        ]
        reset_optimizer = False

    return phases, reset_optimizer


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    params = get_params()

    exp_dir     = params.exp_dir
    model_type  = params.model
    training    = params.training

    ntrain      = params.ntrain
    ntest       = params.ntest
    modes       = params.modes
    width       = params.width
    batch_size  = params.bsize

    epochs          = params.epochs
    learning_rate   = params.learning_rate
    scheduler_step  = params.scheduler_step
    scheduler_gamma = params.scheduler_gamma

    sub       = params.sub
    S         = params.S
    T_in      = params.T_in
    T         = params.T
    step      = params.step
    nu        = params.nu
    rf        = params.rf
    orders_v  = params.orders_v
    n_samples = params.n_samples
    offset    = params.offset
    seed      = params.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data paths ────────────────────────────────────────────────────────────
    is_kolmogorov = (exp_dir == 'kolmogorov')
    is_kh         = (exp_dir == 'kelvin-helmholtz')
    DATA_PATH = ('ns_data_Kolmogorov.mat'    if is_kolmogorov
                 else 'simulation_dataset.mat' if is_kh
                 else f'ns_data_{nu}.mat')

    # ── Output paths ──────────────────────────────────────────────────────────
    training_tag = {
        'multistage':    'ms',
        'multistage_nr': 'ms_nr',
        'singlestage':   'ss',
        'supervised':    'sup',
    }[training]
    ep_label = {
        'multistage':    epochs,
        'multistage_nr': epochs,
        'singlestage':   params.ss_epochs,
        'supervised':    params.sup_epochs,
    }[training]

    path          = f'{model_type}_{training_tag}_{ntrain}_Adam_ep{ep_label}_visc{nu}_m{modes}_w{width}_T{T}'
    path_model    = os.path.join('model', path)
    path_loss_dir = os.path.join('loss',  path)
    os.makedirs(path_loss_dir, exist_ok=True)
    os.makedirs('model', exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    reader  = MatReader(DATA_PATH)
    train_a = reader.read_field('u')[:ntrain,  ::sub, ::sub, offset:offset + T_in]
    train_u = reader.read_field('u')[:ntrain,  ::sub, ::sub, offset + T_in:offset + T + T_in]
    test_a  = reader.read_field('u')[-ntest:,  ::sub, ::sub, offset:offset + T_in]
    test_u  = reader.read_field('u')[-ntest:,  ::sub, ::sub, offset + T_in:offset + T + T_in]

    train_a = train_a.reshape(ntrain, S, S, T_in)
    test_a  = test_a.reshape(ntest,  S, S, T_in)

    # For KH: load ground-truth velocity if available (used for exact residual)
    # Shape: (N, S, S, T_in+T) — same slicing as density
    kh_has_velocity = False
    if is_kh:
        try:
            train_vx = reader.read_field('vx')[:ntrain, ::sub, ::sub,
                            offset + T_in : offset + T_in + T].to(device)
            train_vy = reader.read_field('vy')[:ntrain, ::sub, ::sub,
                            offset + T_in : offset + T_in + T].to(device)
            kh_has_velocity = True
            print("[KH] Ground-truth velocity fields loaded — using exact advection residual.")
        except Exception:
            print("[KH] No velocity fields in dataset — falling back to Biot-Savart residual.")

    gridx = torch.linspace(0, 1, S, device=device).view(1, S, 1, 1).repeat(1, 1, S, 1)
    gridy = torch.linspace(0, 1, S, device=device).view(1, 1, S, 1).repeat(1, S, 1, 1)

    train_a = torch.cat([gridx.repeat(ntrain, 1, 1, 1),
                         gridy.repeat(ntrain, 1, 1, 1),
                         train_a.to(device)], dim=-1)
    test_a  = torch.cat([gridx.repeat(ntest,  1, 1, 1),
                         gridy.repeat(ntest,  1, 1, 1),
                         test_a.to(device)], dim=-1)

    # PINO, ResNet and TF-Net use shuffle=True; spline models use shuffle=False
    # A seeded Generator ensures reproducible batch order even with shuffle=True
    _g = torch.Generator()
    _g.manual_seed(seed)
    pino_shuffle = (model_type in ('pino', 'resnet', 'tfnet'))
    if is_kh and kh_has_velocity:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_a, train_u.to(device),
                                           train_vx, train_vy),
            batch_size=batch_size, shuffle=pino_shuffle,
            generator=_g if pino_shuffle else None)
    else:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_a, train_u.to(device)),
            batch_size=batch_size, shuffle=pino_shuffle,
            generator=_g if pino_shuffle else None)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u.to(device)),
        batch_size=batch_size, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    if model_type == 'phisfno':
        model = Net2d(modes, width).to(device)
    elif model_type == 'pino':
        model = PINONet2d(modes, width).to(device)
    elif model_type == 'unet':
        # For KH (density field, non-zero mean): remove zero-mean centering so the
        # model can represent the density offset (~1.5).  tanh is kept in both cases
        # to bound the spline coefficients and prevent autoregressive divergence.
        model = fluid_model(orders_v,
                            zero_mean=not is_kh,
                            use_tanh=True).to(device)
    elif model_type == 'resnet':
        model = ResNet18_NS(T_in=T_in, width=params.resnet_width).to(device)
    elif model_type == 'tfnet':
        model = TFNet_NS(T_in=T_in, width=params.tfnet_width, fusion_depth=params.tfnet_depth).to(device)

    n_params = sum(reduce(operator.mul, p.size()) for p in model.parameters())
    print(f"[train_model] exp={exp_dir} | model={model_type} | "
          f"training={training} | nu={nu} | params={n_params}")
    if is_kolmogorov and abs(nu - 2e-3) > 1e-10:
        print(f"[WARNING] Il dataset ns_data_Kolmogorov.mat è stato generato con visc=2e-3. "
              f"Stai usando nu={nu}: il residuo fisico sarà calcolato con la viscosità sbagliata.")


    # ── Loss functions ────────────────────────────────────────────────────────
    # LpLossSafe used for UNet eval metric to avoid NaN-induced exploding losses
    loss_data = LpLossSafe(size_average=True) if model_type == 'unet' else LpLoss(size_average=True)
    loss_bd   = LpLoss(size_average=True)

    # ── Common setup ──────────────────────────────────────────────────────────
    # Spline decoding applies only to phisfno and unet (not pino / resnet / tfnet)
    use_spline    = model_type in ('phisfno', 'unet')
    is_supervised = (training == 'supervised')

    v_size  = int(np.prod([i + 1 for i in orders_v]))
    pad     = 1  # 1-pixel boundary
    dataset  = Dataset(batch_size, v_size, S, rf, n_samples, pad) if use_spline else None
    # Fixed border mask used by non-spline models (pino, resnet, tfnet)
    bm_const = make_border_mask(S, device) if not use_spline else None

    # Forcing term  (different for NS and Kolmogorov)
    x_grid = torch.linspace(0, 1, S + 1, device=device)[:-1]
    X, Y   = torch.meshgrid(x_grid, x_grid, indexing='ij')
    if is_kh:
        f = torch.zeros(S, S, device=device)                # KH: no body force (Euler)
    elif is_kolmogorov:
        n_force = 2
        f = -2.0 * torch.cos(2 * math.pi * n_force * Y)    # Kolmogorov: -2·cos(4πy)
    else:
        f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) +
                   torch.cos(2 * math.pi * (X + Y)))        # Navier-Stokes
    f0 = f.unsqueeze(0)  # (1, S, S)
    # dt between consecutive snapshots:
    #   NS / Kolmogorov: 0.25  (original dataset spacing)
    #   KH: tEnd=2.0 / t_out_steps=20 = 0.1
    dt = 0.1 if is_kh else 0.25

    dealias = make_dealias_mask(S, device).unsqueeze(0)   # (1, S, S)
    n_steps = T // step

    # ── Training phases ───────────────────────────────────────────────────────
    phases, reset_optimizer = build_phases(
        training, exp_dir, epochs, params.ss_epochs, params.sup_epochs, model_type)

    # Initial optimizer (shared across phases when reset_optimizer=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase loop
    # ══════════════════════════════════════════════════════════════════════════
    for phase in phases:
        print(f"\n=== Phase '{phase['name']}' | epochs {phase['epochs']} ===\n")

        if reset_optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

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
            for batch in train_loader:
                if is_kh and kh_has_velocity:
                    xx, yy, vx_gt, vy_gt = batch
                else:
                    xx, yy = batch
                    vx_gt = vy_gt = None
                xx, yy = toCuda([xx, yy])
                optimizer.zero_grad()

                # Build boundary / interior masks for this batch
                if use_spline:
                    v_mask        = toCuda(dataset.ask())           # (B, S, S)
                    interior_mask = (1 - v_mask).unsqueeze(1)       # (B, 1, S, S)
                    bm_bool       = (v_mask[0] > 0).reshape(-1)     # (S*S,) bool
                else:
                    # PINO: fixed 1-pixel border interior mask (B, S, S)
                    bd = torch.zeros(batch_size, S, S, device=device)
                    bd[:, :pad,  :] = 1.0
                    bd[:, -pad:, :] = 1.0
                    bd[:, :, :pad]  = 1.0
                    bd[:, :, -pad:] = 1.0
                    interior_mask = (1.0 - bd)   # (B, S, S)

                loss_total = 0.0

                # ── Autoregressive time loop ───────────────────────────────
                for t in range(0, T, step):
                    y  = yy[..., t:t + step]   # (B, S, S, 1)
                    im = model(xx)              # (B, 9, S, S) or (B, S, S, 1)

                    # Decode ω (and velocity for regularisation)
                    if use_spline:
                        ω, v_field, grad_ω, lap_ω = interpolate_states(
                            im,
                            offset=torch.tensor([0.0, 0.0], device=device),
                            orders_v=orders_v)            # ω: (B, S, S, 1)
                        omega_now  = ω[..., 0]            # (B, S, S)
                        omega_prev = xx[..., -1]          # (B, S, S)
                    else:                                  # PINO: ω is im directly
                        ω          = im                   # (B, S, S, 1)
                        v_field    = None
                        omega_now  = im[..., 0]           # (B, S, S)
                        omega_prev = xx[..., -1]

                    # Boundary (supervised) loss
                    if is_supervised:
                        # Full-field Lp — upper-bound baseline, no masking
                        loss_sup_step = loss_bd(
                            ω.reshape(ω.size(0), -1),
                            y.reshape(y.size(0), -1))
                    elif use_spline:
                        loss_sup_step = lp_on_mask(loss_bd, ω, y, bm_bool)
                    else:  # PINO unsupervised boundary
                        bm = bm_const.repeat(im.shape[0], 1, 1, 1)
                        loss_sup_step = loss_bd(
                            (im * bm).reshape(im.shape[0], -1),
                            (y  * bm).reshape(y.shape[0],  -1))

                    # Physics residual (skip if λ_res == 0 to save compute)
                    if λ_res > 0.0:
                        dρ_dt = (omega_now - omega_prev) / dt

                        if is_kh and kh_has_velocity:
                            # ── Full compressible continuity: ∂ρ/∂t + ∇·(ρv) = 0 ─────
                            # Uses ground-truth vx/vy from the dataset.
                            # Conservative divergence form handles ρ∇·v correctly
                            # (pure advection ∂ρ/∂t + v·∇ρ = 0 misses that term at Ma≈0.25).
                            t_idx = t // step  # current time index in the prediction horizon
                            vx_t  = vx_gt[:, :, :, t_idx]   # (B, S, S)
                            vy_t  = vy_gt[:, :, :, t_idx]   # (B, S, S)
                            r = dρ_dt + spectral_div(omega_now * vx_t,
                                                     omega_now * vy_t)
                            if use_spline:
                                res = r.unsqueeze(1)
                                wgt = interior_mask
                            else:
                                res = r
                                wgt = interior_mask

                        elif is_kh and use_spline:
                            # ── Full compressible continuity with spline velocity ───────
                            # ∂ρ/∂t + ∇·(ρ v_spline) = 0
                            vx_s = v_field[..., 0]   # (B, S, S)
                            vy_s = v_field[..., 1]   # (B, S, S)
                            r    = dρ_dt + spectral_div(omega_now * vx_s,
                                                        omega_now * vy_s)
                            res = r.unsqueeze(1)
                            wgt = interior_mask

                        else:
                            # ── Spectral approach ─────────────────────────────────────
                            # NS/Kolmogorov: ∂ω/∂t + u·∇ω = ν∇²ω + f
                            # KH fallback:  ∂ρ/∂t + u_BiotSavart·∇ρ = 0
                            _, _, _, _, lapw, conv = spectral_ops(omega_now, dealias)
                            r = dρ_dt + conv - nu * lapw - f0.expand(xx.size(0), -1, -1)
                            if use_spline:
                                res = r.unsqueeze(1)
                                wgt = interior_mask
                            else:
                                res = r
                                wgt = interior_mask

                        loss_res_step = (res**2 * wgt).sum() / wgt.sum().clamp_min(1)
                    else:
                        loss_res_step = torch.tensor(0.0, device=device)

                    # Spline regularisation (not applied to PINO or supervised)
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

                    # In supervised mode use teacher forcing during training:
                    # the next input receives the ground-truth frame, not the model rollout.
                    next_frame = y if is_supervised else ω

                    # Autoregressive roll
                    pred = ω if t == 0 else torch.cat([pred, ω], dim=-1)
                    xx   = torch.cat([gridx.repeat(batch_size, 1, 1, 1),
                                      gridy.repeat(batch_size, 1, 1, 1),
                                      xx[..., 2 + step:], next_frame], dim=-1)

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
                    for t in range(0, T, step):
                        y  = yy[..., t:t + step]
                        im = model(xx)
                        if use_spline:
                            omega, *_ = interpolate_states(
                                im,
                                offset=torch.tensor([0.0, 0.0], device=device),
                                orders_v=orders_v)
                        else:
                            omega = im

                        test_step_sum += loss_data(
                            omega.reshape(batch_size, -1),
                            y.reshape(batch_size, -1)).item()

                        pred_t = omega if t == 0 else torch.cat([pred_t, omega], dim=-1)
                        xx = torch.cat([gridx.repeat(batch_size, 1, 1, 1),
                                        gridy.repeat(batch_size, 1, 1, 1),
                                        xx[..., 2 + step:], omega], dim=-1)

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
        idx  = list(range(1, phase['epochs'] + 1))
        data = {'train_bd':   train_bd_hist,
                'train_full': train_full_hist,
                'test_full':  test_full_hist,
                'test_step':  test_step_hist}
        if phase['record']:
            data['train_res'] = train_res_hist
        df = pd.DataFrame(data, index=idx)
        df.index.name = 'epoch_in_phase'
        csv_path = os.path.join(path_loss_dir, f"loss_{phase['name']}.csv")
        df.to_csv(csv_path)
        print(f"→ Saved: {csv_path}")

    # ── Save final model ───────────────────────────────────────────────────────
    torch.save(model, path_model)
    print(f"→ Model saved: {path_model}")
