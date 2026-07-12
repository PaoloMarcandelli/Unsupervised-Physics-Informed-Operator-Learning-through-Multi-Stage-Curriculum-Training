"""
visualize.py — Animate KH predictions as a 3-panel GIF.

Loads a trained model from kelvin-helmholtz/model/, runs autoregressive
inference on a test sample, and saves a GIF with three panels:
  - Prediction  ρ̂(x,y,t)
  - Ground truth ρ(x,y,t)
  - Absolute error |ρ̂ - ρ|

Usage
-----
# From Navier_Stokes/
python kelvin-helmholtz/visualize.py --model phisfno --training multistage
python kelvin-helmholtz/visualize.py --model pino    --training supervised --sample_idx 3
python kelvin-helmholtz/visualize.py --model resnet  --training multistage_nr --out kh_resnet.gif
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from scipy.io import loadmat

# ── Path setup ────────────────────────────────────────────────────────────────
# visualize.py lives inside kelvin-helmholtz/
_here       = os.path.dirname(os.path.abspath(__file__))   # kelvin-helmholtz/
_kh_dir     = _here
_ns_root    = os.path.dirname(_here)                       # Navier_Stokes/
_shared_dir = os.path.join(_ns_root, 'shared')
sys.path.insert(0, _shared_dir)

from spline_models import interpolate_states
from models import (Net2d, fluid_model, ResNet18_NS, TFNet_NS,
                    PINONet2d, _PINOSpectralConv2d)  # noqa: F401 — needed for pickle

# Legacy checkpoints saved with torch.save(model) reference _SpectralConv2d
# and PINONet2d as belonging to __main__ (train_model.py at save time).
# Re-exporting them here under their old names lets pickle find them.
import __main__
__main__._SpectralConv2d = _PINOSpectralConv2d
__main__.PINONet2d       = PINONet2d


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Visualize KH model predictions as GIF')
    p.add_argument('--model',    choices=['phisfno', 'pino', 'unet', 'resnet', 'tfnet'],
                   required=True)
    p.add_argument('--training', choices=['multistage', 'multistage_nr',
                                          'singlestage', 'supervised'],
                   required=True)
    # Match the hyperparams used at training time
    p.add_argument('--ntrain',      type=int,   default=160)
    p.add_argument('--ntest',       type=int,   default=40)
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--ss_epochs',   type=int,   default=200)
    p.add_argument('--sup_epochs',  type=int,   default=150)
    p.add_argument('--nu',          type=float, default=0.0)
    p.add_argument('--modes',       type=int,   default=12)
    p.add_argument('--width',       type=int,   default=20)
    p.add_argument('--sub',         type=int,   default=2)
    p.add_argument('--S',           type=int,   default=64)
    p.add_argument('--T_in',        type=int,   default=10)
    p.add_argument('--T',           type=int,   default=10)
    p.add_argument('--offset',      type=int,   default=0)
    p.add_argument('--orders_v',    type=int,   nargs=2, default=[2, 2])
    p.add_argument('--sample_idx',  type=int,   default=0,
                   help='Which test sample to visualize (0-indexed)')
    p.add_argument('--fps',         type=int,   default=4,
                   help='Frames per second in the output GIF')
    p.add_argument('--out',         type=str,   default=None,
                   help='Output GIF path (default: auto-named in kelvin-helmholtz/)')
    return p.parse_args()


# ── Model path ────────────────────────────────────────────────────────────────

def build_model_path(args):
    tag = {'multistage': 'ms', 'multistage_nr': 'ms_nr',
           'singlestage': 'ss', 'supervised': 'sup'}[args.training]
    ep  = {'multistage':    args.epochs,
           'multistage_nr': args.epochs,
           'singlestage':   args.ss_epochs,
           'supervised':    args.sup_epochs}[args.training]
    name = (f'{args.model}_{tag}_{args.ntrain}_Adam_ep{ep}'
            f'_visc{args.nu}_m{args.modes}_w{args.width}_T{args.T}')
    return os.path.join(_kh_dir, 'model', name)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_sample(args, device):
    """Load one test sample.  Returns:
        a  : (1, S, S, 2+T_in)   — model input (gridx, gridy, density stack)
        u  : (1, S, S, T)        — ground truth future frames
    """
    data_path = os.path.join(_kh_dir, 'simulation_dataset.mat')
    raw = loadmat(data_path)['u']          # (N, 128, 128, T_total)  float

    sub    = args.sub
    S      = args.S
    T_in   = args.T_in
    T      = args.T
    offset = args.offset
    ntest  = args.ntest
    idx    = args.sample_idx

    if idx >= ntest:
        raise ValueError(f'--sample_idx {idx} >= --ntest {ntest}')

    # Same slicing as train_model.py
    a_np = raw[-ntest + idx : -ntest + idx + 1,
               ::sub, ::sub,
               offset : offset + T_in]         # (1, S, S, T_in)
    u_np = raw[-ntest + idx : -ntest + idx + 1,
               ::sub, ::sub,
               offset + T_in : offset + T_in + T]  # (1, S, S, T)

    a = torch.tensor(a_np, dtype=torch.float32, device=device)
    u = torch.tensor(u_np, dtype=torch.float32, device=device)

    gridx = torch.linspace(0, 1, S, device=device).view(1, S, 1, 1).repeat(1, 1, S, 1)
    gridy = torch.linspace(0, 1, S, device=device).view(1, 1, S, 1).repeat(1, S, 1, 1)

    a = torch.cat([gridx, gridy, a], dim=-1)   # (1, S, S, 2+T_in)
    return a, u


# ── Autoregressive inference ──────────────────────────────────────────────────

def predict(model, a, u, args, device):
    """Run autoregressive rollout.  Returns pred (S,S,T) and gt (S,S,T)."""
    use_spline = args.model in ('phisfno', 'unet')
    orders_v   = args.orders_v
    T          = args.T
    step       = 1
    S          = args.S

    gridx = torch.linspace(0, 1, S, device=device).view(1, S, 1, 1).repeat(1, 1, S, 1)
    gridy = torch.linspace(0, 1, S, device=device).view(1, 1, S, 1).repeat(1, S, 1, 1)

    xx     = a.clone()
    frames = []

    model.eval()
    grad_ctx = torch.enable_grad if use_spline else torch.no_grad
    with grad_ctx():
        for t in range(0, T, step):
            im = model(xx)

            if use_spline:
                omega, *_ = interpolate_states(
                    im,
                    offset=torch.tensor([0.0, 0.0], device=device),
                    orders_v=orders_v)
            else:
                omega = im   # (1, S, S, 1)

            frames.append(omega[0, :, :, 0].detach().cpu().numpy())   # (S, S)

            xx = torch.cat([gridx, gridy,
                            xx[..., 2 + step:], omega.detach()], dim=-1)

    pred = np.stack(frames, axis=-1)          # (S, S, T)
    gt   = u[0].cpu().numpy()                 # (S, S, T)
    return pred, gt


# ── GIF creation ─────────────────────────────────────────────────────────────

def make_gif(pred, gt, out_path, fps, model_name, training_name, sample_idx):
    T   = pred.shape[-1]
    err = np.abs(pred - gt)

    # Shared colour ranges
    vmin = min(pred.min(), gt.min())
    vmax = max(pred.max(), gt.max())
    emax = err.max()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f'Kelvin-Helmholtz  |  {model_name.upper()} {training_name}  |  test sample {sample_idx}',
        fontsize=12, fontweight='bold')

    titles = ['Prediction  $\\hat{\\rho}$',
              'Ground truth  $\\rho$',
              'Absolute error  $|\\hat{\\rho}-\\rho|$']
    cmaps  = ['viridis', 'viridis', 'inferno']

    ims = []
    for ax, title, cmap in zip(axes, titles, cmaps):
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        im = ax.imshow(np.zeros((pred.shape[0], pred.shape[1])),
                       origin='lower', cmap=cmap,
                       vmin=(0 if cmap == 'inferno' else vmin),
                       vmax=(emax if cmap == 'inferno' else vmax),
                       animated=True)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

    time_text = fig.text(0.5, 0.01, '', ha='center', fontsize=10)

    def update(frame_idx):
        ims[0].set_data(pred[:, :, frame_idx].T)
        ims[1].set_data(gt[:, :, frame_idx].T)
        ims[2].set_data(err[:, :, frame_idx].T)
        time_text.set_text(f'Step  {frame_idx + 1} / {T}')
        return ims + [time_text]

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 // fps, blit=True)

    ani.save(out_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f'GIF saved → {out_path}')


# ── Snapshot image ────────────────────────────────────────────────────────────

def save_snapshot(pred, gt, out_path, model_name, training_name, sample_idx, T_in):
    """Save a 3×3 static image: rows = time snapshots, cols = pred / gt / error."""
    T   = pred.shape[-1]
    err = np.abs(pred - gt)

    # Three snapshot indices: first, middle, last prediction
    snap_indices = [0, T // 2, T - 1]
    snap_labels  = [T_in + i + 1 for i in snap_indices]   # absolute time steps

    vmin = min(pred.min(), gt.min())
    vmax = max(pred.max(), gt.max())
    emax = err.max()

    col_titles = ['Prediction  $\\hat{\\rho}$',
                  'Ground truth  $\\rho$',
                  'Absolute error  $|\\hat{\\rho}-\\rho|$']
    cmaps      = ['viridis', 'viridis', 'inferno']

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    fig.suptitle(
        f'Kelvin-Helmholtz  |  {model_name.upper()} {training_name}  |  test sample {sample_idx}',
        fontsize=13, fontweight='bold')

    for row, (idx, label) in enumerate(zip(snap_indices, snap_labels)):
        for col, (cmap, col_title) in enumerate(zip(cmaps, col_titles)):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(col_title, fontsize=11)

            data = [pred[:, :, idx], gt[:, :, idx], err[:, :, idx]][col]
            im = ax.imshow(data.T, origin='lower', cmap=cmap,
                           vmin=(0     if cmap == 'inferno' else vmin),
                           vmax=(emax  if cmap == 'inferno' else vmax))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
            ax.set_ylabel(f't = {label}', fontsize=10)
            if col == 0:
                ax.set_ylabel(f't = {label}', fontsize=10, labelpad=4)
            if col == 2:
                mean_err = err[:, :, idx].mean()
                ax.text(0.5, -0.08,
                        rf'Mean $|\hat{{\rho}}-\rho|$ = {mean_err:.3e}',
                        ha='center', va='top',
                        transform=ax.transAxes, fontsize=9,
                        clip_on=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Snapshot saved → {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ─────────────────────────────────────────────────────────────
    model_path = build_model_path(args)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f'Model not found: {model_path}\n'
            f'Run train_model.py first with matching hyperparameters.')
    print(f'Loading model: {model_path}')
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f'Loading test sample {args.sample_idx} from simulation_dataset.mat ...')
    a, u = load_test_sample(args, device)

    # ── Inference ──────────────────────────────────────────────────────────────
    print('Running autoregressive rollout ...')
    pred, gt = predict(model, a, u, args, device)

    l2_err = np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12)
    print(f'Relative L2 error: {l2_err:.4f}')

    # ── Build output path ──────────────────────────────────────────────────────
    if args.out is None:
        out_dir  = os.path.join(_kh_dir, 'animations', args.model)
        out_path = os.path.join(out_dir, f'{args.training}.gif')
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # ── Make GIF ───────────────────────────────────────────────────────────────
    print('Rendering GIF ...')
    make_gif(pred, gt, out_path,
             fps=args.fps,
             model_name=args.model,
             training_name=args.training,
             sample_idx=args.sample_idx)

    # ── Save snapshot image ────────────────────────────────────────────────────
    snap_path = out_path.replace('.gif', '_snapshots.png')
    save_snapshot(pred, gt, snap_path,
                  model_name=args.model,
                  training_name=args.training,
                  sample_idx=args.sample_idx,
                  T_in=args.T_in)


if __name__ == '__main__':
    main()
