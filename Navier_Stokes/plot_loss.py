"""
plot_loss.py — unified loss-curve plotting for NS-vorticity and Kolmogorov.

Usage
-----
python plot_loss.py --exp_dir navier-stokes-vorticity --nu 1e-3 --offset 40
python plot_loss.py --exp_dir navier-stokes-vorticity --nu 1e-4 --offset 20
python plot_loss.py --exp_dir kolmogorov              --nu 2e-3 --offset 30
"""

import argparse
import os
import sys
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 20,
})

GREEN  = "#00E676"
BLUE   = "#0062B2"
PURPLE = "#7C4DFF"
BLACK  = "#000000"

# Comparison plot — one colour per model
MODEL_COLORS = {
    'phisfno': "#00E676",   # green
    'pino':    "#0062B2",   # blue
    'resnet':  "#7C4DFF",   # purple
    'tfnet':   "#FF6D00",   # orange
    'unet':    "#D50000",   # red
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_params():
    p = argparse.ArgumentParser(description="Plot loss curves for one experiment")
    p.add_argument('--exp_dir',
                   choices=['navier-stokes-vorticity', 'kolmogorov', 'kelvin-helmholtz'],
                   required=True)
    p.add_argument('--ntrain',     type=int,   default=16)
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--ss_epochs',  type=int,   default=150)
    p.add_argument('--sup_epochs', type=int,   default=150)
    p.add_argument('--nu',         type=float, default=1e-3)
    p.add_argument('--modes',      type=int,   default=12)
    p.add_argument('--width',      type=int,   default=20)
    p.add_argument('--T',          type=int,   default=10)
    p.add_argument('--offset',     type=int,   default=40)
    return p.parse_args()


# ── Stats helpers ─────────────────────────────────────────────────────────────

def final_plateau_stats(y, rel_tol=2e-3, min_len=20, tail_max=100):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.nan, np.nan, 0, 0
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
    i0 = n - length
    seg = y[i0:]
    return float(np.mean(seg)), float(np.std(seg)), i0, n


def rolling_stats(y, win=11):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y, y
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=float) / win
    mean = np.convolve(ypad, kernel, mode='valid')
    sq   = np.convolve(ypad**2, kernel, mode='valid')
    std  = np.sqrt(np.maximum(0.0, sq - mean**2))
    return mean, std


def rolling_stats_log(y, win=11, eps=1e-12):
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, eps)
    m_log, s_log = rolling_stats(np.log10(y), win=win)
    return 10**m_log, 10**(m_log - s_log), 10**(m_log + s_log)


# ── Phase / path helpers ──────────────────────────────────────────────────────

def infer_phase_names(training, loss_dir):
    if training in ('multistage', 'multistage_nr'):
        # NS has 5 phases (with boundary_only_2); Kolmogorov and KH have 3
        if os.path.isfile(os.path.join(loss_dir, 'loss_boundary_only_2.csv')):
            return ['boundary_only_1', 'boundary_only_2',
                    'full_loss_1', 'full_loss_2', 'full_loss_3']
        return ['boundary_only_1', 'full_loss_1', 'full_loss_2']
    if training == 'singlestage':
        return ['full_loss_1']
    if training == 'supervised':
        return ['supervised']
    raise ValueError(f"Unknown training strategy: {training}")


def build_run_name(model_type, training, params):
    tag = {'multistage': 'ms', 'multistage_nr': 'ms_nr',
           'singlestage': 'ss', 'supervised': 'sup'}[training]
    ep  = {'multistage':    params.epochs,
           'multistage_nr': params.epochs,
           'singlestage':   params.ss_epochs,
           'supervised':    params.sup_epochs}[training]
    return (f'{model_type}_{tag}_{params.ntrain}_Adam_ep{ep}'
            f'_visc{params.nu}_m{params.modes}_w{params.width}_T{params.T}')


def resolve_loss_dir(loss_root, model_type, training, params):
    exact = os.path.join(loss_root, build_run_name(model_type, training, params))
    if os.path.isdir(exact):
        return exact
    # Legacy name variants
    model_variants    = {'unet': ['unet', 'UNet']}.get(model_type, [model_type])
    _tag_map = {'multistage': 'ms', 'multistage_nr': 'ms_nr',
                'singlestage': 'ss', 'supervised': 'sup'}
    _legacy  = {'supervised': ['sup', 'supervised', 'superv']}
    training_variants = _legacy.get(training, [_tag_map[training]])
    candidates = []
    for mv in model_variants:
        for tv in training_variants:
            pat = (f"{mv}_{tv}_{params.ntrain}_Adam_ep*"
                   f"_visc{params.nu}_m{params.modes}_w{params.width}_T{params.T}")
            candidates.extend(glob(os.path.join(loss_root, pat)))
    candidates = sorted(set(candidates))
    if candidates:
        def epoch_key(p):
            try:
                return int(os.path.basename(p).split('_Adam_ep', 1)[1].split('_', 1)[0])
            except (IndexError, ValueError):
                return -1
        return max(candidates, key=epoch_key)
    return exact


# ── Data loader ───────────────────────────────────────────────────────────────

def load_test_full_series(loss_dir, training, win=11):
    phases = infer_phase_names(training, loss_dir)
    xs, ys, phase_lengths = [], [], []
    cum_epoch = 0
    for phase in phases:
        csv_path = os.path.join(loss_dir, f'loss_{phase}.csv')
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path, index_col='epoch_in_phase')
        if 'test_full' not in df.columns:
            continue
        y = df['test_full'].to_numpy()
        xs.append(np.arange(1, len(y) + 1) + cum_epoch)
        ys.append(y)
        phase_lengths.append(len(y))
        cum_epoch += len(y)
    if not xs:
        return None, None, None
    return np.concatenate(xs), np.concatenate(ys), phase_lengths


# ── Per-model plot ────────────────────────────────────────────────────────────

def plot_model_family(model_type, title, params, loss_root):
    training_order = [
        ('multistage',    'M-S',                   GREEN),
        ('multistage_nr', 'M-S (no reset)',         BLUE),
        ('singlestage',   r'S-S $\lambda_{res}=1$', PURPLE),
        ('supervised',    'Supervised',             BLACK),
    ]

    plt.figure(figsize=(8, 5))
    plt.yscale('log')
    ax = plt.gca()
    plotted_any = False
    split_positions = None
    ms_loss_dir = resolve_loss_dir(loss_root, model_type, 'multistage', params)

    for training, label, color in training_order:
        loss_dir = resolve_loss_dir(loss_root, model_type, training, params)
        x, y, phase_lengths = load_test_full_series(loss_dir, training, win=11)
        if x is None:
            print(f"  [skip] {model_type}/{training}: no data in {loss_dir}")
            continue

        ymean, ylow, yup = rolling_stats_log(y, win=11)
        ax.plot(x, ymean, label=label, color=color, linewidth=2.0, zorder=3)
        ax.fill_between(x, np.maximum(ylow, 1e-12), yup,
                        color=color, alpha=0.18, linewidth=0, zorder=2)
        plotted_any = True

        mean, std, i0, i1 = final_plateau_stats(y)
        print(f"  {title} ({training:15s}) plateau: {mean:.3e} +/- {std:.3e}  [window {i1-i0}]")

        if training == 'multistage' and phase_lengths is not None:
            split_positions = np.cumsum(phase_lengths)[:-1]

    if not plotted_any:
        plt.close()
        return

    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$L_2$')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    if split_positions is not None:
        for xpos in split_positions:
            ax.axvline(xpos, color='k', linewidth=0.6, dashes=(2, 2), alpha=0.9)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 0.06 * (xmax - xmin))

    os.makedirs(ms_loss_dir, exist_ok=True)
    out_png = os.path.join(ms_loss_dir, 'multi_vs_single_errbar_log.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"  Saved: {out_png}")


# ── Cross-model comparison plot (multistage with reset) ───────────────────────

def plot_ms_comparison(model_specs, params, loss_root):
    """Plot the multistage-with-reset loss for every model on a single graph.

    The figure is saved to  <exp_dir>/loss_comparison/ms_comparison_nu<nu>.png
    """
    comp_dir = os.path.join(os.path.dirname(loss_root), 'loss_comparison')
    os.makedirs(comp_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_yscale('log')

    split_positions = None   # use the first model that has phase info

    for model_type, title in model_specs:
        color    = MODEL_COLORS.get(model_type, BLACK)
        loss_dir = resolve_loss_dir(loss_root, model_type, 'multistage', params)
        x, y, phase_lengths = load_test_full_series(loss_dir, 'multistage', win=11)
        if x is None:
            print(f"  [skip comparison] {model_type}/multistage: no data in {loss_dir}")
            continue

        ymean, ylow, yup = rolling_stats_log(y, win=11)
        ax.plot(x, ymean, label=title, color=color, linewidth=2.0, zorder=3)
        ax.fill_between(x, np.maximum(ylow, 1e-12), yup,
                        color=color, alpha=0.18, linewidth=0, zorder=2)

        mean, std, i0, i1 = final_plateau_stats(y)
        print(f"  [comparison] {title:12s} (multistage) plateau: "
              f"{mean:.3e} +/- {std:.3e}  [window {i1-i0}]")

        if split_positions is None and phase_lengths is not None:
            split_positions = np.cumsum(phase_lengths)[:-1]

    if not ax.lines:
        plt.close()
        print("  [comparison] No data found — skipping comparison plot.")
        return

    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$L_2$')
    ax.legend(loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    if split_positions is not None:
        for xpos in split_positions:
            ax.axvline(xpos, color='k', linewidth=0.6, dashes=(2, 2), alpha=0.9)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 0.06 * (xmax - xmin))

    nu_str  = f"{params.nu:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
    out_png = os.path.join(comp_dir, f'ms_comparison_nu{nu_str}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"  Saved comparison: {out_png}")


# ── Per-phase convergence table (PhIS-FNO multistage, NS only) ───────────────

def print_phisfno_phase_convergence(params, loss_root):
    """
    For PhIS-FNO multistage, print a convergence table with one row per phase.
    Each row shows the plateau mean (last stable window) of:
        L_bd   — boundary loss  (train_bd)
        L_res  — residual loss  (train_res)  → '-' when λ_res = 0
        L_train— full train loss (train_full)
        L_test — validation loss (test_full)

    Uses the same final_plateau_stats() algorithm as seed_stats.py.
    """
    loss_dir = resolve_loss_dir(loss_root, 'phisfno', 'multistage', params)
    phase_names = infer_phase_names('multistage', loss_dir)

    nu_str = f"{params.nu:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
    has_residual = {'full_loss_1', 'full_loss_2', 'full_loss_3'}

    # column widths
    cw = [5, 22, 12, 12, 12, 12]
    header = (f"{'Ph':<{cw[0]}}{'Phase name':<{cw[1]}}"
              f"{'L_bd':>{cw[2]}}{'L_res':>{cw[3]}}"
              f"{'L_train':>{cw[4]}}{'L_test':>{cw[5]}}")
    sep = '─' * sum(cw)

    print(f"\n{'═'*sum(cw)}")
    print(f"  PhIS-FNO multistage — convergence plateau  (ν = {nu_str})")
    print(f"  Loss dir: {loss_dir}")
    print(f"{'═'*sum(cw)}")
    print(header)
    print(sep)

    def _plateau(df, col):
        if col not in df.columns:
            return None
        y = df[col].dropna().to_numpy()
        if len(y) == 0:
            return None
        if np.all(y == 0):
            return None
        mean, *_ = final_plateau_stats(y)
        return mean

    def _fmt(val):
        if val is None:
            return f"{'—':>12}"
        return f"{val:>12.4e}"

    for ph_idx, phase in enumerate(phase_names, start=1):
        csv_path = os.path.join(loss_dir, f'loss_{phase}.csv')
        if not os.path.isfile(csv_path):
            print(f"  Ph {ph_idx}  {phase:<20}  [CSV not found]")
            continue

        df = pd.read_csv(csv_path, index_col='epoch_in_phase')

        l_bd    = _plateau(df, 'train_bd')
        l_res   = _plateau(df, 'train_res') if phase in has_residual else None
        l_train = _plateau(df, 'train_full')
        l_test  = _plateau(df, 'test_full')

        print(f"{ph_idx:<{cw[0]}}{phase:<{cw[1]}}"
              f"{_fmt(l_bd)}{_fmt(l_res)}{_fmt(l_train)}{_fmt(l_test)}")

    print(sep)


# ── Per-phase detail plots (PhIS-FNO multistage, NS only) ────────────────────

def plot_phisfno_phases(params, loss_root):
    """
    For PhIS-FNO multistage only, save one figure per phase showing:
      Phases 1-2 (boundary_only): boundary (blue), train (orange), validation (green)
      Phases 3-5 (full_loss):     + residual (red)

    Figures are saved to  <exp_dir>/figures/nu<nu_str>/phase_<n>.png
    """
    C_VALID = '#00C853'   # green  — validation (test_full)
    C_BD    = '#0062B2'   # blue   — boundary   (train_bd)
    C_TRAIN = '#FF6D00'   # orange — train full  (train_full)
    C_RES   = '#D50000'   # red    — residual    (train_res)

    loss_dir = resolve_loss_dir(loss_root, 'phisfno', 'multistage', params)

    phase_names = infer_phase_names('multistage', loss_dir)
    if not phase_names:
        print("  [phases] No phase CSVs found for phisfno/multistage.")
        return

    nu_str  = f"{params.nu:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
    fig_dir = os.path.join(os.path.dirname(loss_root), 'figures', f'nu{nu_str}')
    os.makedirs(fig_dir, exist_ok=True)

    phase_titles = {
        'boundary_only_1': 'Phase 1 — Boundary Only',
        'boundary_only_2': 'Phase 2 — Boundary Only',
        'full_loss_1':     'Phase 3 — Full Loss',
        'full_loss_2':     'Phase 4 — Full Loss',
        'full_loss_3':     'Phase 5 — Full Loss',
        # 3-phase variant (Kolmogorov / KH)
        'full_loss_2_3p':  'Phase 3 — Full Loss',
    }
    has_residual = {'full_loss_1', 'full_loss_2', 'full_loss_3'}

    for ph_idx, phase in enumerate(phase_names, start=1):
        csv_path = os.path.join(loss_dir, f'loss_{phase}.csv')
        if not os.path.isfile(csv_path):
            print(f"  [phases] Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path, index_col='epoch_in_phase')
        epochs = np.arange(1, len(df) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_yscale('log')

        def _plot(col, color, label):
            if col not in df.columns:
                return
            y = df[col].to_numpy()
            if np.all(y == 0) or np.all(np.isnan(y)):
                return
            ymean, ylow, yup = rolling_stats_log(y, win=11)
            ax.plot(epochs, ymean, color=color, linewidth=2.0, label=label, zorder=3)
            ax.fill_between(epochs, np.maximum(ylow, 1e-12), yup,
                            color=color, alpha=0.18, linewidth=0, zorder=2)

        _plot('train_bd',   C_BD,    'Boundary loss')
        _plot('train_full', C_TRAIN, 'Train loss')
        _plot('test_full',  C_VALID, 'Validation loss')
        if phase in has_residual:
            _plot('train_res', C_RES, 'Residual loss')

        title_str = phase_titles.get(phase, f'Phase {ph_idx}')
        ax.set_title(
            rf'PhIS-FNO — {title_str}  ($\nu={nu_str}$)',
            fontsize=18)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$L_2$')
        ax.legend(loc='best')
        ax.grid(True, which='both', ls=':', alpha=0.5)

        out_png = os.path.join(fig_dir, f'phase_{ph_idx}.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"  Saved: {out_png}")


# ── Resolution invariance plot ────────────────────────────────────────────────

def plot_resolution_invariance(model_specs, params, base_dir):
    """Load the best (multistage) model for each architecture and evaluate
    the relative L2 error at resolutions 32, 64, 128, 256.

    The figure is saved to  <exp_dir>/figures/resolution_<nu>.png
    """
    shared_dir = os.path.join(base_dir, 'shared')
    exp_dir    = os.path.join(base_dir, params.exp_dir)

    for d in (shared_dir, exp_dir):
        if d not in sys.path:
            sys.path.insert(0, d)

    try:
        from utilities3 import LpLoss, MatReader
        from spline_models import interpolate_states
        from models import Net2d, fluid_model, ResNet18_NS, TFNet_NS, PINONet2d
    except ImportError as e:
        print(f"  [resolution] Cannot import shared utilities: {e}")
        return

    # Models were saved when train_model.py ran as __main__, so pickle stored
    # every class defined in that module (including helpers) as '__main__.ClassName'.
    # Inject the full contents of every shared module into __main__ so torch.load works.
    import __main__, importlib
    for _mod_name in ('models', 'spline_models', 'operators', 'utilities3', 'unet_parts'):
        try:
            _mod = importlib.import_module(_mod_name)
            for _name, _obj in vars(_mod).items():
                if not _name.startswith('__'):
                    setattr(__main__, _name, _obj)
        except ImportError:
            pass
    # Legacy alias: checkpoints saved before the class was renamed
    if hasattr(__main__, '_PINOSpectralConv2d') and not hasattr(__main__, '_SpectralConv2d'):
        __main__._SpectralConv2d = __main__._PINOSpectralConv2d

    is_kolmogorov = (params.exp_dir == 'kolmogorov')
    data_file = (os.path.join(exp_dir, 'ns_data_Kolmogorov.mat') if is_kolmogorov
                 else os.path.join(exp_dir, f'ns_data_{params.nu}.mat'))

    if not os.path.isfile(data_file):
        print(f"  [resolution] Data file not found: {data_file}")
        return

    import torch
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn  = LpLoss(size_average=True)

    RESOLUTIONS = [32, 64, 128, 256]
    S_FULL  = 256       # raw dataset resolution
    T_IN    = 10        # input sequence length (matches training default)
    T       = params.T
    step    = 1
    offset  = params.offset
    ntest   = 2
    orders_v = [2, 2]

    model_root = os.path.join(exp_dir, 'model')
    results    = {}     # model_type -> list[float]

    for model_type, title in model_specs:
        model_path = os.path.join(model_root,
                                  build_run_name(model_type, 'multistage', params))
        if not os.path.isfile(model_path):
            print(f"  [resolution] Model not found: {model_path}")
            continue

        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
            # Patch attributes that may be missing in older checkpoints
            if model_type == 'unet':
                if not hasattr(model, 'use_tanh'):
                    model.use_tanh = True
                if not hasattr(model, 'zero_mean'):
                    model.zero_mean = True
            model.eval()
        except Exception as e:
            print(f"  [resolution] Could not load {model_type}: {e}")
            continue

        use_spline = model_type in ('phisfno', 'unet')
        errors = []

        for res in RESOLUTIONS:
            sub = S_FULL // res
            S   = res

            reader = MatReader(data_file)
            test_a = reader.read_field('u')[-ntest:, ::sub, ::sub,
                                           offset:offset + T_IN]
            test_u = reader.read_field('u')[-ntest:, ::sub, ::sub,
                                           offset + T_IN:offset + T_IN + T]
            test_a = test_a.reshape(ntest, S, S, T_IN)

            gridx = (torch.linspace(0, 1, S, device=device)
                     .view(1, S, 1, 1).repeat(1, 1, S, 1))
            gridy = (torch.linspace(0, 1, S, device=device)
                     .view(1, 1, S, 1).repeat(1, S, 1, 1))

            xx     = torch.cat([gridx.repeat(ntest, 1, 1, 1),
                                 gridy.repeat(ntest, 1, 1, 1),
                                 test_a.to(device)], dim=-1)
            test_u = test_u.to(device)

            offset_t = torch.tensor([0.0, 0.0], device=device)

            try:
                with torch.no_grad():
                    pred_t = None
                    for t in range(0, T, step):
                        im = model(xx)
                        if use_spline:
                            omega, *_ = interpolate_states(
                                im, offset=offset_t, orders_v=orders_v)
                        else:
                            omega = im

                        pred_t = omega if pred_t is None else torch.cat(
                            [pred_t, omega], dim=-1)
                        xx = torch.cat([gridx.repeat(ntest, 1, 1, 1),
                                        gridy.repeat(ntest, 1, 1, 1),
                                        xx[..., 2 + step:], omega], dim=-1)

                    err = loss_fn(pred_t.reshape(ntest, -1),
                                  test_u.reshape(ntest, -1)).item()
            except Exception as e:
                print(f"  [resolution] {title} res={res}: error during eval — {e}")
                err = float('nan')

            errors.append(err)
            print(f"  [resolution] {title:12s}  res={res:3d}: L2 = {err:.4e}")

        results[model_type] = (title, errors)

    if not results:
        print("  [resolution] No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_type, (title, errors) in results.items():
        color  = MODEL_COLORS.get(model_type, BLACK)
        valid  = [(r, e) for r, e in zip(RESOLUTIONS, errors)
                  if not (isinstance(e, float) and np.isnan(e))]
        if not valid:
            continue
        rs, es = zip(*valid)
        ax.plot(rs, es, marker='o', color=color, linewidth=2.0, label=title)

    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xticks(RESOLUTIONS)
    ax.set_xticklabels([str(r) for r in RESOLUTIONS])
    ax.set_xlabel('Resolution (grid size)')
    ax.set_ylabel('Relative error')
    nu_str = f"{params.nu:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
    ax.legend(loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    fig_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_png = os.path.join(fig_dir, f'resolution_{nu_str}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"  Saved: {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    params    = get_params()
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    loss_root = os.path.join(base_dir, params.exp_dir, 'loss')

    model_specs = [
        ('pino',    'PINO'),
        ('phisfno', 'PhIS-FNO'),
        ('unet',    'UNet'),
        ('resnet',  'ResNet-18'),
        ('tfnet',   'TF-Net'),
    ]

    print(f"\n=== Plotting losses for exp_dir={params.exp_dir}, nu={params.nu} ===\n")
    for model_type, title in model_specs:
        plot_model_family(model_type, title, params, loss_root)

    print(f"\n=== Plotting multi-stage comparison for exp_dir={params.exp_dir}, nu={params.nu} ===\n")
    plot_ms_comparison(model_specs, params, loss_root)

    print(f"\n=== Plotting resolution invariance for exp_dir={params.exp_dir}, nu={params.nu} ===\n")
    plot_resolution_invariance(model_specs, params, base_dir)

    if params.exp_dir == 'navier-stokes-vorticity':
        nu_str = f"{params.nu:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
        print(f"\n=== Per-phase convergence table (PhIS-FNO MS, nu={nu_str}) ===")
        print_phisfno_phase_convergence(params, loss_root)
        print(f"\n=== Plotting per-phase detail (PhIS-FNO MS, nu={nu_str}) ===\n")
        plot_phisfno_phases(params, loss_root)


if __name__ == '__main__':
    main()
