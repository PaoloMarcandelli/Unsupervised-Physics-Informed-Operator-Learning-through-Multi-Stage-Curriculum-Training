"""
finitevolume.py — Kelvin-Helmholtz instability dataset generator.

Solves the 2-D compressible Euler equations via a finite-volume Muscl-Hancock
scheme (Rusanov flux). Each simulation starts from the same base shear-layer
initial condition perturbed by a different random seed, producing independent
trajectories of the density field ρ(x,y,t).

Output .mat layout (compatible with train_model.py):
    field 'u'  shape  (N_samples, S, S, T_total)
    where T_total = T_out  (default 20)
    → use offset=0, T_in=10, T=10  in train_model.py

Usage:
    python finitevolume.py --nsamples 200 --out simulation_dataset.mat
    python finitevolume.py --nsamples 200 --nworkers 4 --out simulation_dataset.mat
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')           # no display needed — works in WSL2 / headless

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat


# ── Finite-volume primitives ──────────────────────────────────────────────────

def getConserved(rho, vx, vy, P, gamma, vol):
    Mass   = rho * vol
    Momx   = rho * vx * vol
    Momy   = rho * vy * vol
    Energy = (P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2)) * vol
    return Mass, Momx, Momy, Energy


def getPrimitive(Mass, Momx, Momy, Energy, gamma, vol):
    rho = Mass / vol
    vx  = Momx  / (rho * vol)
    vy  = Momy  / (rho * vol)
    P   = (Energy / vol - 0.5 * rho * (vx**2 + vy**2)) * (gamma - 1)
    return rho, vx, vy, P


def getGradient(f, dx):
    f_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    f_dy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
    f_XL = np.roll(f - f_dx * dx / 2, -1, axis=0)
    f_XR = f + f_dx * dx / 2
    f_YL = np.roll(f - f_dy * dx / 2, -1, axis=1)
    f_YR = f + f_dy * dx / 2
    return f_XL, f_XR, f_YL, f_YR


def applyFluxes(F, flux_X, flux_Y, dx, dt):
    F -= dt * dx * flux_X
    F += dt * dx * np.roll(flux_X, 1, axis=0)
    F -= dt * dx * flux_Y
    F += dt * dx * np.roll(flux_Y, 1, axis=1)
    return F


def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2)
    en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2)

    rho_star  = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star   = 0.5 * (en_L  + en_R)
    P_star    = (gamma - 1) * (en_star - 0.5 * (momx_star**2 + momy_star**2) / rho_star)

    flux_Mass   = momx_star
    flux_Momx   = momx_star**2 / rho_star + P_star
    flux_Momy   = momx_star * momy_star / rho_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star

    C_L = np.sqrt(gamma * P_L / rho_L) + np.abs(vx_L)
    C_R = np.sqrt(gamma * P_R / rho_R) + np.abs(vx_R)
    C   = np.maximum(C_L, C_R)

    flux_Mass   -= C * 0.5 * (rho_L - rho_R)
    flux_Momx   -= C * 0.5 * (rho_L * vx_L  - rho_R * vx_R)
    flux_Momy   -= C * 0.5 * (rho_L * vy_L  - rho_R * vy_R)
    flux_Energy -= C * 0.5 * (en_L  - en_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy


# ── Single simulation ─────────────────────────────────────────────────────────

def run_simulation(seed=0, perturbation_strength=0.05, t_out_steps=20):
    """
    Run one KH simulation and return density snapshots.

    Parameters
    ----------
    seed : int
        RNG seed — fixes the random initial perturbation.
    perturbation_strength : float
        Amplitude of the random noise added to the base IC.
    t_out_steps : int
        Number of evenly-spaced output snapshots in [0, tEnd].

    Returns
    -------
    np.ndarray, shape (N, N, t_out_steps)
        Density field ρ(x, y, t) at each output time.
    """
    rng = np.random.default_rng(seed)

    # ── Grid ──────────────────────────────────────────────────────────────────
    N        = 128
    boxsize  = 1.0
    gamma    = 5.0 / 3.0
    courant  = 0.4
    t        = 0.0
    tEnd     = 2.0
    dx       = boxsize / N
    vol      = dx**2

    xlin = np.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    Y, X = np.meshgrid(xlin, xlin)

    # ── Base initial condition (shear layer) ──────────────────────────────────
    w0    = 0.1
    sigma = 0.05 / np.sqrt(2.0)
    inner = np.abs(Y - 0.5) < 0.25

    rho = 1.0 + inner.astype(float)
    vx  = -0.5 + inner.astype(float)
    vy  = w0 * np.sin(4 * np.pi * X) * (
            np.exp(-(Y - 0.25)**2 / (2 * sigma**2)) +
            np.exp(-(Y - 0.75)**2 / (2 * sigma**2)))
    P   = 2.5 * np.ones_like(X)

    # ── Random perturbation (reproducible via seed) ───────────────────────────
    rho += perturbation_strength * rng.standard_normal(rho.shape)
    vx  += perturbation_strength * rng.standard_normal(vx.shape)
    vy  += perturbation_strength * rng.standard_normal(vy.shape)
    P   += perturbation_strength * rng.standard_normal(P.shape)

    # clip to avoid negative pressure / density
    rho = np.maximum(rho, 1e-6)
    P   = np.maximum(P,   1e-6)

    Mass, Momx, Momy, Energy = getConserved(rho, vx, vy, P, gamma, vol)

    # ── Time integration ──────────────────────────────────────────────────────
    output_times  = np.linspace(0.0, tEnd, t_out_steps + 1)[1:]  # exclude t=0
    output_fields = []
    next_out      = 0

    while next_out < t_out_steps:
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, gamma, vol)

        # CFL time step
        c_sound = np.sqrt(gamma * P / rho)
        dt = courant * np.min(dx / (c_sound + np.sqrt(vx**2 + vy**2)))

        # Snap dt to next output time if close
        t_target = output_times[next_out]
        if t + dt >= t_target:
            dt = t_target - t

        # Gradients
        rho_dx, rho_dy = getGradient(rho, dx)
        vx_dx,  vx_dy  = getGradient(vx,  dx)
        vy_dx,  vy_dy  = getGradient(vy,  dx)
        P_dx,   P_dy   = getGradient(P,   dx)

        # Half-step predictor
        rho_p = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_p  = vx  - 0.5 * dt * (vx * vx_dx  + vy * vx_dy  + (1.0 / rho) * P_dx)
        vy_p  = vy  - 0.5 * dt * (vx * vy_dx  + vy * vy_dy  + (1.0 / rho) * P_dy)
        P_p   = P   - 0.5 * dt * (gamma * P * (vx_dx + vy_dy) + vx * P_dx + vy * P_dy)

        rho_p = np.maximum(rho_p, 1e-6)
        P_p   = np.maximum(P_p,   1e-6)

        # Face extrapolation
        rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_p, rho_dx, rho_dy, dx)
        vx_XL,  vx_XR,  vx_YL,  vx_YR  = extrapolateInSpaceToFace(vx_p,  vx_dx,  vx_dy,  dx)
        vy_XL,  vy_XR,  vy_YL,  vy_YR  = extrapolateInSpaceToFace(vy_p,  vy_dx,  vy_dy,  dx)
        P_XL,   P_XR,   P_YL,   P_YR   = extrapolateInSpaceToFace(P_p,   P_dx,   P_dy,   dx)

        # Rusanov fluxes
        fMx, fMomxX, fMomyX, fEX = getFlux(rho_XL, rho_XR, vx_XL,  vx_XR,  vy_XL,  vy_XR,  P_XL,  P_XR,  gamma)
        fMy, fMomyY, fMomxY, fEY = getFlux(rho_YL, rho_YR, vy_YL,  vy_YR,  vx_YL,  vx_YR,  P_YL,  P_YR,  gamma)

        Mass   = applyFluxes(Mass,   fMx,    fMy,    dx, dt)
        Momx   = applyFluxes(Momx,   fMomxX, fMomxY, dx, dt)
        Momy   = applyFluxes(Momy,   fMomyX, fMomyY, dx, dt)
        Energy = applyFluxes(Energy, fEX,    fEY,    dx, dt)

        t += dt

        if abs(t - t_target) < 1e-12 * tEnd:
            output_fields.append((rho.copy(), vx.copy(), vy.copy()))
            next_out += 1

    # shapes: each entry is (rho, vx, vy) each (N, N)
    rhos = np.stack([f[0] for f in output_fields], axis=-1)   # (N, N, T)
    vxs  = np.stack([f[1] for f in output_fields], axis=-1)   # (N, N, T)
    vys  = np.stack([f[2] for f in output_fields], axis=-1)   # (N, N, T)
    return rhos, vxs, vys


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(nsamples, t_out_steps, perturbation_strength, seed_offset, nworkers):
    """
    Generate `nsamples` independent KH trajectories.

    Returns three arrays of shape  (nsamples, N, N, t_out_steps):
        rho_all, vx_all, vy_all
    Saved as fields 'u', 'vx', 'vy' in the .mat file.
    train_model.py uses field 'u' (density) for supervised/boundary loss
    and 'vx'/'vy' for the physics residual when available.
    """
    from multiprocessing import Pool

    args = [
        (seed_offset + i, perturbation_strength, t_out_steps)
        for i in range(nsamples)
    ]

    if nworkers > 1:
        with Pool(processes=nworkers) as pool:
            results = pool.starmap(run_simulation, args)
    else:
        results = []
        for idx, a in enumerate(args):
            if idx % 10 == 0:
                print(f"  sim {idx + 1}/{nsamples} ...")
            results.append(run_simulation(*a))

    rho_all = np.stack([r[0] for r in results], axis=0)   # (N, N, N, T)
    vx_all  = np.stack([r[1] for r in results], axis=0)
    vy_all  = np.stack([r[2] for r in results], axis=0)
    return rho_all, vx_all, vy_all


# ── Diagnostic plot ───────────────────────────────────────────────────────────

def plot_sample(dataset, out_path, n_frames=6):
    """Save a grid of density snapshots from the first trajectory."""
    traj = dataset[0]          # (N, N, T)
    T    = traj.shape[-1]
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 3))
    for ax, ti in zip(axes, idxs):
        im = ax.imshow(traj[:, :, ti].T, origin='lower', cmap='viridis')
        ax.set_title(f't={ti}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle('KH density  ρ(x,y,t)  — sample 0', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Diagnostic plot saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='KH dataset generator')
    p.add_argument('--nsamples',   type=int,   default=200,
                   help='Total number of simulations (default 200)')
    p.add_argument('--t_out',      type=int,   default=20,
                   help='Number of output snapshots per sim (default 20)')
    p.add_argument('--perturb',    type=float, default=0.05,
                   help='Random perturbation amplitude (default 0.05)')
    p.add_argument('--seed_offset',type=int,   default=0,
                   help='First RNG seed (default 0)')
    p.add_argument('--nworkers',   type=int,   default=1,
                   help='Parallel workers via multiprocessing (default 1)')
    p.add_argument('--out',        type=str,   default='simulation_dataset.mat',
                   help='Output .mat filename')
    p.add_argument('--plot',       action='store_true',
                   help='Save a diagnostic grid plot alongside the .mat file')
    return p.parse_args()


def main():
    args = get_args()

    print(f"\n=== KH dataset generation ===")
    print(f"  nsamples   = {args.nsamples}")
    print(f"  t_out      = {args.t_out}  (use offset=0, T_in=10, T=10 in train_model.py)")
    print(f"  perturb    = {args.perturb}")
    print(f"  nworkers   = {args.nworkers}")
    print(f"  output     = {args.out}\n")

    rho_all, vx_all, vy_all = generate_dataset(
        nsamples=args.nsamples,
        t_out_steps=args.t_out,
        perturbation_strength=args.perturb,
        seed_offset=args.seed_offset,
        nworkers=args.nworkers,
    )

    print(f"\nDataset shape: {rho_all.shape}   dtype: {rho_all.dtype}")
    print(f"  ρ   min={rho_all.min():.4f}  max={rho_all.max():.4f}  mean={rho_all.mean():.4f}")
    print(f"  vx  min={vx_all.min():.4f}  max={vx_all.max():.4f}")
    print(f"  vy  min={vy_all.min():.4f}  max={vy_all.max():.4f}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # 'u'  → density ρ  (used by train_model.py for supervised/boundary loss)
    # 'vx' → x-velocity (used for exact physics residual ∂ρ/∂t + v·∇ρ = 0)
    # 'vy' → y-velocity
    savemat(args.out, {
        'u':  rho_all.astype(np.float32),
        'vx': vx_all.astype(np.float32),
        'vy': vy_all.astype(np.float32),
    })
    print(f"Saved → {args.out}")

    if args.plot:
        plot_out = args.out.replace('.mat', '_sample.png')
        plot_sample(rho_all, plot_out)


if __name__ == '__main__':
    main()
