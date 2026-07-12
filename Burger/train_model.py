"""
train_model.py — Unified Burger 1-D training script.

Replaces burger_phisfno_ms.py, burger_phisfno_ms_nr.py,
burger_phisfno_ss.py, burger_supervised.py.
Adds PINO-FD and PINO-Spectral as new baselines.

Usage examples
--------------
python train_model.py --model phisfno       --training multistage
python train_model.py --model phisfno       --training multistage_nr
python train_model.py --model phisfno       --training singlestage
python train_model.py --model phisfno       --training supervised
python train_model.py --model pino_fd       --training multistage
python train_model.py --model pino_spectral --training multistage

Arguments
---------
--model     phisfno       — FNO 1D with cubic splines (3 coefficients/node)
            pino_fd       — FNO 1D with central finite differences (O(h²))
            pino_spectral — FNO 1D with spectral (FFT) derivatives
                            (expected to diverge at high resolution due to
                             k_max^2 amplification of noise)
--training  multistage    — 3 phases, Adam reset at each phase boundary
            multistage_nr — 3 phases, shared Adam (no reset)
            singlestage   — 1 phase, full loss from the start
            supervised    — full-domain L2 supervision (upper bound)
"""

import os, math, json
from timeit import default_timer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Local dependencies (already present in the Burger directory) ──────────────
from utilities3 import MatReader, LpLoss
from spline_models1d import interpolate_states   # used by phisfno
from models import Net1d, SpectralConv1d         # Net1d: (B,3,N) spline output


# ════════════════════════════════════════════════════════════════════════════
# 1.  PINO-FD / PINO-Spectral model  (direct u output, 1 channel)
#     Both share the same FNO backbone — only the derivative in the residual
#     differs. pino_fd uses central FD O(h²); pino_spectral uses FFT.
# ════════════════════════════════════════════════════════════════════════════

class FNO1d(nn.Module):
    """
    FNO backbone shared by pino_fd and pino_spectral.
    Same architecture as Net1d (models.py) but the head outputs u directly
    (1 channel) instead of 3 spline coefficients.
    input  : (B, N, 2)   [u_0, x_grid]
    output : (B, N)      [u prediction]
    """
    def __init__(self, modes, width):
        super().__init__()
        self.fc0   = nn.Linear(2, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0    = nn.Conv1d(width, width, 1)
        self.w1    = nn.Conv1d(width, width, 1)
        self.w2    = nn.Conv1d(width, width, 1)
        self.w3    = nn.Conv1d(width, width, 1)
        self.bn0   = nn.BatchNorm1d(width)
        self.bn1   = nn.BatchNorm1d(width)
        self.bn2   = nn.BatchNorm1d(width)
        self.bn3   = nn.BatchNorm1d(width)
        self.fc1   = nn.Linear(width, 128)
        self.fc2   = nn.Linear(128, 1)          # 1 channel: direct u

    def forward(self, x):
        # x: (B, N, 2)
        x = self.fc0(x).permute(0, 2, 1)        # (B, width, N)
        x = F.relu(self.bn0(self.conv0(x) + self.w0(x)))
        x = F.relu(self.bn1(self.conv1(x) + self.w1(x)))
        x = F.relu(self.bn2(self.conv2(x) + self.w2(x)))
        x =        self.bn3(self.conv3(x) + self.w3(x))
        x = x.permute(0, 2, 1)                  # (B, N, width)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                          # (B, N, 1)
        return x.squeeze(-1)                     # (B, N)

    def count_params(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())


# ════════════════════════════════════════════════════════════════════════════
# 2.  Derivative operators
# ════════════════════════════════════════════════════════════════════════════

# ── 2a.  Finite differences  (periodic, O(h²)) ───────────────────────────────

def fd_first(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Central O(h²) first derivative, periodic BC. u: (B, S)."""
    ux = torch.empty_like(u)
    ux[:, 1:-1] = (u[:, 2:]  - u[:, :-2]) / (2.0 * dx)
    ux[:, 0]    = (u[:, 1]   - u[:, -1])  / (2.0 * dx)
    ux[:, -1]   = (u[:, 0]   - u[:, -2])  / (2.0 * dx)
    return ux

def fd_second(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Central O(h²) second derivative, periodic BC. u: (B, S)."""
    uxx = torch.empty_like(u)
    uxx[:, 1:-1] = (u[:, 2:]  - 2.0 * u[:, 1:-1] + u[:, :-2]) / dx**2
    uxx[:, 0]    = (u[:, 1]   - 2.0 * u[:, 0]    + u[:, -1])  / dx**2
    uxx[:, -1]   = (u[:, 0]   - 2.0 * u[:, -1]   + u[:, -2])  / dx**2
    return uxx


# ── 2b.  Spectral derivatives via FFT  (periodic domain [0, L)) ──────────────
# WARNING: at S=2048, k_max ≈ 1024 → k² ≈ 10^6.
# Any high-frequency noise in u is multiplied by -(2πk/L)² ≈ -4×10^7,
# making the unsupervised residual explode. This is intentional: these
# functions exist to demonstrate the instability, not to fix it.

def spectral_first(u: torch.Tensor, L: float) -> torch.Tensor:
    """
    Spectral first derivative d/dx via rfft, periodic [0, L). u: (B, S).
    Nyquist mode zeroed out for anti-aliasing (standard in pseudospectral).
    """
    N    = u.shape[-1]
    u_hat = torch.fft.rfft(u, dim=-1)                         # (B, N//2+1) complex
    k     = torch.arange(u_hat.shape[-1], device=u.device,
                         dtype=torch.float32)
    wavenumber = (2.0 * math.pi / L) * k
    if N % 2 == 0:
        wavenumber[N // 2] = 0.0   # zero Nyquist (avoids asymmetry)
    ux_hat = 1j * wavenumber * u_hat
    return torch.fft.irfft(ux_hat, n=N, dim=-1)

def spectral_second(u: torch.Tensor, L: float) -> torch.Tensor:
    """
    Spectral second derivative d²/dx² via rfft, periodic [0, L). u: (B, S).
    """
    N    = u.shape[-1]
    u_hat = torch.fft.rfft(u, dim=-1)
    k     = torch.arange(u_hat.shape[-1], device=u.device,
                         dtype=torch.float32)
    wavenumber = (2.0 * math.pi / L) * k
    if N % 2 == 0:
        wavenumber[N // 2] = 0.0
    uxx_hat = -(wavenumber ** 2) * u_hat
    return torch.fft.irfft(uxx_hat, n=N, dim=-1)


# ════════════════════════════════════════════════════════════════════════════
# 3.  Mask & loss helpers
# ════════════════════════════════════════════════════════════════════════════

def lp_on_mask_1d(myloss, x, y, mask_bool):
    """LpLoss restricted to boundary indices. x,y: (B,S) or (B,S,1)."""
    if x.dim() == 3: x = x.squeeze(-1)
    if y.dim() == 3: y = y.squeeze(-1)
    X = x.reshape(x.size(0), -1)
    Y = y.reshape(y.size(0), -1)
    return myloss(X[:, mask_bool], Y[:, mask_bool])

def build_boundary_mask_1d(S, k_bd, device):
    """Bool mask (S,) — True on k_bd left and right points."""
    m = torch.zeros(S, dtype=torch.bool, device=device)
    k = min(k_bd, S // 2)
    m[:k] = True;  m[-k:] = True
    return m


# ════════════════════════════════════════════════════════════════════════════
# 4.  Adam diagnostic utilities  (from burger_phisfno_ms.py)
# ════════════════════════════════════════════════════════════════════════════

def _flatten_abs(t: torch.Tensor):
    x = t.detach()
    if x.is_complex(): x = x.abs()
    return x.reshape(-1).abs()

def _robust_stats_abs(t: torch.Tensor):
    a = _flatten_abs(t).float().cpu()
    if a.numel() == 0:
        return dict(mean=0., median=0., p75=0.)
    return dict(mean=a.mean().item(), median=a.median().item(), p75=a.quantile(0.75).item())

@torch.no_grad()
def cosine_m_g(optimizer):
    num = den_m = den_g = 0.0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None: continue
            st = optimizer.state.get(p, None)
            if not st or 'exp_avg' not in st: continue
            m = st['exp_avg']; g = p.grad
            num   += (m.conj() * g).real.sum().item()
            den_m += (m.abs()**2).sum().item()
            den_g += (g.abs()**2).sum().item()
    denom = (den_m**0.5) * (den_g**0.5) + 1e-12
    return float(num / denom)

@torch.no_grad()
def adam_eta_term_stats(optimizer):
    terms = []
    for group in optimizer.param_groups:
        beta1, beta2 = group['betas']; eps = group['eps']
        for p in group['params']:
            if p.grad is None: continue
            st = optimizer.state.get(p, None)
            if not st or 'exp_avg' not in st or 'exp_avg_sq' not in st: continue
            m = st['exp_avg']; v = st['exp_avg_sq']
            step = int(st.get('step', 0))
            bc1 = 1.0 - beta1**step if step > 0 else 1.0
            bc2 = 1.0 - beta2**step if step > 0 else 1.0
            term = _flatten_abs(m / bc1) / (_flatten_abs(v / bc2).sqrt() + eps)
            terms.append(term)
    if not terms:
        return {'mean': 0., 'median': 0., 'p95': 0., 'max': 0.}
    T = torch.cat(terms).float().cpu()
    return {'mean': T.mean().item(), 'median': T.median().item(),
            'p95': T.quantile(0.95).item(), 'max': T.max().item()}


# ════════════════════════════════════════════════════════════════════════════
# 5.  Stage-switch diagnostics  (only active for multistage)
# ════════════════════════════════════════════════════════════════════════════

class StageDiagnostics:
    def __init__(self, csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self._cache = None
        if not os.path.exists(csv_path):
            import csv
            with open(csv_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'stage_from', 'stage_to', 'layer',
                    'beta1', 'beta2', 'lr', 'eps',
                    'delta_median', 'sigma_median', 'gnew_median',
                    'ratio_num', 'ratio_den', 'R_lb', 'timestamp'])

    @staticmethod
    def _bias_correct(state, beta1, beta2, step):
        bc1 = 1.0 - beta1**step if step > 0 else 1.0
        bc2 = 1.0 - beta2**step if step > 0 else 1.0
        return state['exp_avg'] / bc1, state['exp_avg_sq'] / bc2

    @torch.no_grad()
    def stage_end_snapshot(self, model, optimizer):
        cache = {}
        group = optimizer.param_groups[0]
        beta1, beta2 = group['betas']
        for name, p in model.named_parameters():
            st = optimizer.state.get(p, None)
            if st is None or 'exp_avg' not in st: continue
            step = int(st.get('step', 0))
            m_hat, v_hat = self._bias_correct(st, beta1, beta2, step)
            cache[name] = {
                'delta_median': _robust_stats_abs(m_hat)['median'],
                'sigma_median': _robust_stats_abs(v_hat.sqrt())['median']
            }
        self._cache = cache

    @torch.no_grad()
    def stage_switch_measure(self, stage_from, stage_to, model, optimizer):
        if not self._cache: return
        import csv, time
        group  = optimizer.param_groups[0]
        beta1, beta2 = group['betas']
        lr = group.get('lr', 1e-3); eps = group.get('eps', 1e-8)
        rows = []
        for name, p in model.named_parameters():
            if p.grad is None or name not in self._cache: continue
            g_med   = _robust_stats_abs(p.grad)['median']
            delta   = self._cache[name]['delta_median']
            sigma   = self._cache[name]['sigma_median']
            r_num   = (beta1 * delta) / ((1.0 - beta1) * (g_med + 1e-12))
            r_den   = (math.sqrt(1.0 - beta2) * g_med) / (math.sqrt(beta2) * (sigma + 1e-12))
            denom   = math.sqrt(1.0 - beta2) * ((1.0 - beta1) * g_med + beta1 * delta) + 1e-12
            R_lb    = (math.sqrt(beta2) * sigma * (1.0 - beta1)) / denom
            rows.append([stage_from, stage_to, name, beta1, beta2, lr, eps,
                         delta, sigma, g_med, r_num, r_den, R_lb, time.time()])
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerows(rows)
        self._cache = None


# ════════════════════════════════════════════════════════════════════════════
# 6.  Phase schedule builder
# ════════════════════════════════════════════════════════════════════════════

def build_phases(training, model_type, epochs, ss_epochs, sup_epochs):
    """
    Returns (phases, reset_optimizer, reset_scheduler).

    reset_optimizer : True  → recreate Adam + scheduler at every phase (multistage)
    reset_scheduler : True  → recreate scheduler only, keep Adam moments (multistage_nr)

    Phase weights mirror the individual existing scripts exactly.
    Both multistage and multistage_nr use the same λ weights; the only
    difference is whether Adam state is carried across phases.

    Spline regularization (λ_smooth, λ_tv, λ_mean) only for phisfno.
    """
    lam_reg = {'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2} \
              if model_type == 'phisfno' else {}

    if training == 'multistage':
        # mirrors burger_phisfno_ms.py — Adam reset at each phase
        phases = [
            {'name': 'full_loss_1', 'record': True, 'epochs': epochs,
             'λ_sup': 0.8, 'λ_res': 0.5, **lam_reg},
            {'name': 'full_loss_2', 'record': True, 'epochs': epochs,
             'λ_sup': 0.5, 'λ_res': 1.0, **lam_reg},
            {'name': 'full_loss_3', 'record': True, 'epochs': epochs,
             'λ_sup': 0.5, 'λ_res': 1.5, **lam_reg},
        ]
        reset_optimizer = True
        reset_scheduler = True   # implied by optimizer reset

    elif training == 'multistage_nr':
        # mirrors burger_phisfno_ms_nr.py — same λ weights as multistage,
        # but Adam moments carry across phases; only the scheduler resets
        # (in the original, StepLR is re-created inside the phase loop).
        phases = [
            {'name': 'full_loss_1', 'record': True, 'epochs': epochs,
             'λ_sup': 0.8, 'λ_res': 0.5, **lam_reg},
            {'name': 'full_loss_2', 'record': True, 'epochs': epochs,
             'λ_sup': 0.5, 'λ_res': 1.0, **lam_reg},
            {'name': 'full_loss_3', 'record': True, 'epochs': epochs,
             'λ_sup': 0.5, 'λ_res': 1.5, **lam_reg},
        ]
        reset_optimizer = False
        reset_scheduler = True   # LR resets to lr at each phase, moments do not

    elif training == 'singlestage':
        # mirrors burger_phisfno_ss.py
        phases = [
            {'name': 'full_loss_1', 'record': True, 'epochs': ss_epochs,
             'λ_sup': 0.8, 'λ_res': 1.0, **lam_reg},
        ]
        reset_optimizer = False
        reset_scheduler = False

    elif training == 'supervised':
        # mirrors burger_supervised.py — full-domain L2, no physics
        phases = [
            {'name': 'supervised', 'record': True, 'epochs': sup_epochs,
             'λ_sup': 1.0, 'λ_res': 0.0},
        ]
        reset_optimizer = False
        reset_scheduler = False

    else:
        raise ValueError(f'Unknown training strategy: {training}')

    return phases, reset_optimizer, reset_scheduler


# ════════════════════════════════════════════════════════════════════════════
# 7.  Argument parsing
# ════════════════════════════════════════════════════════════════════════════

def get_params():
    import argparse
    p = argparse.ArgumentParser(description='Unified Burger 1-D training')

    # Core
    p.add_argument('--model',    choices=['phisfno', 'pino_fd', 'pino_spectral'],
                   required=True)
    p.add_argument('--training', choices=['multistage', 'multistage_nr',
                                          'singlestage', 'supervised'],
                   required=True)

    # Dataset
    p.add_argument('--data',    default='dataset/burgers1d_dataset.mat')
    p.add_argument('--ntrain',  type=int,   default=90)
    p.add_argument('--ntest',   type=int,   default=10)
    p.add_argument('--sub',     type=int,   default=4,
                   help='Spatial sub-sampling factor (default 4 → S=2048)')
    p.add_argument('--k_bd',    type=int,   default=100,
                   help='Boundary thickness in grid points')

    # Architecture
    p.add_argument('--modes',   type=int,   default=32)
    p.add_argument('--width',   type=int,   default=64)

    # Training schedule
    p.add_argument('--epochs',      type=int,   default=100,
                   help='Epochs per phase for multistage variants')
    p.add_argument('--ss_epochs',   type=int,   default=200,
                   help='Epochs for singlestage')
    p.add_argument('--sup_epochs',  type=int,   default=300,
                   help='Epochs for supervised')
    p.add_argument('--batch_size',  type=int,   default=10)
    p.add_argument('--lr',          type=float, default=1e-2)
    p.add_argument('--sched_step',  type=int,   default=10)
    p.add_argument('--sched_gamma', type=float, default=0.5)

    # Physics
    p.add_argument('--nu',   type=float, default=0.1)
    p.add_argument('--dt',   type=float, default=0.1,
                   help='Time step between IC and target snapshot')
    p.add_argument('--Ldom', type=float, default=2*math.pi)

    p.add_argument('--seed', type=int, default=0)

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# 8.  Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    params = get_params()

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = params.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # cuDNN: force deterministic algorithms; disable auto-tuner
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    def _worker_init(worker_id):
        """Each DataLoader worker gets a deterministic but distinct seed."""
        np.random.seed(seed + worker_id)
    # ─────────────────────────────────────────────────────────────────────────

    model_type   = params.model
    training     = params.training
    use_spline   = (model_type == 'phisfno')
    use_spectral = (model_type == 'pino_spectral')
    is_sup       = (training == 'supervised')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Grid / physics ────────────────────────────────────────────────────────
    nu    = params.nu
    dt    = params.dt
    Ldom  = params.Ldom
    sub   = params.sub
    S     = (2**13) // sub              # e.g. 2048 for sub=4
    dx    = Ldom / S                    # grid spacing

    ntrain     = params.ntrain
    ntest      = params.ntest
    batch_size = params.batch_size
    k_bd       = params.k_bd
    modes      = params.modes
    width      = params.width

    # ── Output paths ──────────────────────────────────────────────────────────
    tag = {'multistage': 'ms', 'multistage_nr': 'ms_nr',
           'singlestage': 'ss', 'supervised': 'sup'}[training]
    ep_label = {'multistage': params.epochs,   'multistage_nr': params.epochs,
                'singlestage': params.ss_epochs, 'supervised': params.sup_epochs}[training]

    path          = f'Burger1d_{model_type}_{tag}_{ntrain}_Adam_ep{ep_label}_visc{nu}_m{modes}_w{width}_s{seed}'
    path_model    = os.path.join('model', path)
    path_loss_dir = os.path.join('loss',  path)
    for d in ('model', path_loss_dir):
        os.makedirs(d, exist_ok=True)

    print(f'[train_model] model={model_type} | training={training} | '
          f'S={S} | nu={nu} | dt={dt} | modes={modes} | width={width}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    reader = MatReader(params.data)
    x_data = reader.read_field('a')[:, ::sub]          # (N, S) initial conditions
    y_data = reader.read_field('u')[:, -1, ::sub]      # (N, S) target snapshot

    x_train = x_data[:ntrain];  y_train = y_data[:ntrain]
    x_test  = x_data[-ntest:];  y_test  = y_data[-ntest:]

    grid    = torch.tensor(np.linspace(0, 2*math.pi, S).reshape(1, S, 1), dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain, S, 1), grid.repeat(ntrain, 1, 1)], dim=2)
    x_test  = torch.cat([x_test.reshape(ntest,  S, 1), grid.repeat(ntest,  1, 1)], dim=2)

    _g = torch.Generator()
    _g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True,
        worker_init_fn=_worker_init, generator=_g)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=batch_size, shuffle=False,
        worker_init_fn=_worker_init)

    # ── Masks ─────────────────────────────────────────────────────────────────
    border_mask   = build_boundary_mask_1d(S, k_bd, device)   # (S,) bool
    interior_mask = ~border_mask

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_type == 'phisfno':
        model = Net1d(modes, width).to(device)
    else:
        # pino_fd and pino_spectral share the same FNO1d backbone;
        # derivative method only differs inside the training loop.
        model = FNO1d(modes, width).to(device)
    print(f'Model parameters: {model.count_params():,}')

    # ── Losses ────────────────────────────────────────────────────────────────
    myloss = LpLoss(size_average=False)

    # ── Phase schedule ────────────────────────────────────────────────────────
    phases, reset_optimizer, reset_scheduler = build_phases(
        training, model_type, params.epochs, params.ss_epochs, params.sup_epochs)

    # Stage diagnostics (meaningful only for multistage with reset)
    stage_diag = StageDiagnostics(os.path.join(path_loss_dir, 'stage_switch.csv'))

    # Initial optimizer (shared when reset_optimizer=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params.sched_step, gamma=params.sched_gamma)

    eta_term_history = []
    epoch_cum        = 0

    # ══════════════════════════════════════════════════════════════════════════
    # Phase loop
    # ══════════════════════════════════════════════════════════════════════════
    for phase_idx, phase in enumerate(phases):
        print(f"\n=== Phase '{phase['name']}' | epochs {phase['epochs']} ===\n")

        if reset_optimizer:
            # multistage: recreate Adam + scheduler (fresh moments, fresh LR)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=params.sched_step, gamma=params.sched_gamma)
        elif reset_scheduler:
            # multistage_nr: keep Adam moments, but reset LR schedule
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=params.sched_step, gamma=params.sched_gamma)

        λ_sup    = phase['λ_sup']
        λ_res    = phase['λ_res']
        λ_smooth = phase.get('λ_smooth', 0.0)
        λ_tv     = phase.get('λ_tv',     0.0)
        λ_mean   = phase.get('λ_mean',   0.0)

        train_bd_hist, train_res_hist, test_hist = [], [], []
        first_batch_in_phase = True

        # ── Epoch loop ────────────────────────────────────────────────────────
        for ep in range(phase['epochs']):
            model.train()
            t1 = default_timer()
            ep_bd = ep_res = 0.0

            # ── Batch loop ────────────────────────────────────────────────────
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)   # (B,S,2), (B,S)
                optimizer.zero_grad(set_to_none=True)

                im = model(x)
                # im: (B,3,S) for phisfno | (B,S) for pino_fd

                # ── Decode u for boundary/supervision loss (at grid nodes) ────
                if use_spline:
                    u_bd, _, _ = interpolate_states(
                        im, offset=torch.tensor([0.0], device=device), orders_v=[2])
                else:
                    u_bd = im   # (B, S)

                # ── Boundary / supervision loss ───────────────────────────────
                if is_sup:
                    loss_bd = myloss(u_bd.reshape(u_bd.size(0), -1),
                                     y.reshape(y.size(0), -1))
                else:
                    loss_bd = lp_on_mask_1d(myloss, u_bd, y, border_mask)

                # ── Physics residual ──────────────────────────────────────────
                if λ_res > 0.0:
                    if use_spline:
                        # Evaluate at midpoints (offset=0.5) for phisfno
                        u_r, ux, uxx = interpolate_states(
                            im, offset=torch.tensor([0.5], device=device), orders_v=[2])
                    elif use_spectral:
                        # Spectral (FFT) derivatives — periodic [0, Ldom)
                        # At S=2048, k_max^2 ≈ 10^6: noise is catastrophically
                        # amplified, which is the expected pathological behaviour.
                        u_r  = im
                        ux   = spectral_first(u_r,  Ldom)
                        uxx  = spectral_second(u_r, Ldom)
                    else:
                        # Central FD O(h²) for pino_fd
                        u_r = im
                        ux  = fd_first(u_r,  dx)
                        uxx = fd_second(u_r, dx)

                    d_u_dt   = (u_r - x[..., 0]) / dt
                    residual  = d_u_dt + u_r * ux - nu * uxx
                    loss_res  = (residual[:, interior_mask]**2).mean()
                else:
                    loss_res = torch.tensor(0.0, device=device)
                    u_r = u_bd   # fallback reference for regularisation

                # ── Spline regularisation (phisfno only, not supervised) ──────
                # Uses u_r (offset=0.5, interior) to match the original scripts:
                # in burger_phisfno_ms.py and ms_nr.py the variable `u` is
                # overwritten by the offset=0.5 call, so regularisation there
                # implicitly runs on midpoint values. When λ_res=0 the else
                # branch above sets u_r = u_bd, so the fallback is consistent.
                smooth_loss = tv_loss = mean_loss = torch.tensor(0.0, device=device)
                if use_spline and not is_sup and λ_smooth > 0.0:
                    ddx         = u_r[:, 1:] - u_r[:, :-1]
                    smooth_loss = ddx.pow(2).mean()
                    tv_loss     = ddx.abs().mean()
                    mean_loss   = u_r.mean(dim=-1).pow(2).mean()

                loss = (λ_sup    * loss_bd
                        + λ_res    * loss_res
                        + λ_smooth * smooth_loss
                        + λ_tv     * tv_loss
                        + λ_mean   * mean_loss)
                loss.backward()

                # Stage-switch measurement (first batch of phase > 0).
                # Works for both reset and no-reset: it only reads whatever
                # optimizer state exists at this point (fresh after reset,
                # accumulated m_{t-1}/v_{t-1} if not reset).
                if first_batch_in_phase and phase_idx > 0:
                    stage_diag.stage_switch_measure(
                        stage_from=phases[phase_idx - 1]['name'],
                        stage_to=phase['name'],
                        model=model, optimizer=optimizer)
                    first_batch_in_phase = False

                optimizer.step()
                ep_bd  += loss_bd.item()
                ep_res += loss_res.item()

            scheduler.step()

            # ── Evaluation ────────────────────────────────────────────────────
            model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    im = model(x)
                    if use_spline:
                        u_pred, _, _ = interpolate_states(
                            im, offset=torch.tensor([0.0], device=device), orders_v=[2])
                    else:
                        u_pred = im
                    test_l2 += myloss(u_pred.reshape(u_pred.size(0), -1),
                                      y.reshape(y.size(0), -1)).item()

            ep_bd   /= ntrain
            ep_res  /= len(train_loader)
            test_l2 /= ntest

            # Adam diagnostics
            stats  = adam_eta_term_stats(optimizer)
            cos_mg = cosine_m_g(optimizer)
            eta_term_history.append({
                'epoch_global':   epoch_cum + ep + 1,
                'phase':          phase['name'],
                'epoch_in_phase': ep + 1,
                'term_mean':      stats['mean'],
                'term_median':    stats['median'],
                'term_p95':       stats['p95'],
                'term_max':       stats['max'],
                'lr':             optimizer.param_groups[0]['lr'],
                'eta_eff_mean':   optimizer.param_groups[0]['lr'] * stats['mean'],
                'train_bd':       ep_bd,
                'train_res':      ep_res,
                'test_full':      test_l2,
                'cos_m_g':        cos_mg,
                'reset_opt':      reset_optimizer,
            })

            elapsed = default_timer() - t1
            print(f'Ep {ep+1:3d}/{phase["epochs"]} │ {elapsed:5.2f}s │ '
                  f'bd {ep_bd:8.4e} │ res {ep_res:8.4e} │ test {test_l2:8.4e} │ '
                  f'η_eff {optimizer.param_groups[0]["lr"]*stats["mean"]:8.3e}')

            train_bd_hist.append(ep_bd)
            train_res_hist.append(ep_res)
            test_hist.append(test_l2)

        # ── End of phase: snapshot for stage diagnostics ──────────────────────
        # Captures m_hat/v_hat of whatever optimizer is currently active,
        # regardless of reset_optimizer, so multistage_nr also gets its
        # (never-reset) moments recorded before the next phase begins.
        stage_diag.stage_end_snapshot(model, optimizer)

        epoch_cum += phase['epochs']

        # ── Save phase CSV ────────────────────────────────────────────────────
        idx = list(range(1, phase['epochs'] + 1))
        df  = pd.DataFrame(
            {'train_bd': train_bd_hist, 'train_res': train_res_hist,
             'test_full': test_hist},
            index=idx)
        df.index.name = 'epoch_in_phase'
        csv_path = os.path.join(path_loss_dir, f"loss_{phase['name']}.csv")
        df.to_csv(csv_path)
        print(f'→ Saved: {csv_path}')

    # ── Save model & Adam diagnostics ─────────────────────────────────────────
    torch.save(model, path_model)
    print(f'→ Model saved: {path_model}')

    eta_df = pd.DataFrame(eta_term_history)
    eta_csv = os.path.join(path_loss_dir, 'eta_term_history.csv')
    eta_df.to_csv(eta_csv, index=False)
    print(f'→ Eta-term history: {eta_csv}')

