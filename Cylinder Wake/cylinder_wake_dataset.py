#!/usr/bin/env python3
"""
Cylinder wake dataset generator (Raissi/Karniadakis-style) — pure Python/NumPy version
====================================================================================

This script generates a 2D cylinder-wake dataset on a **rectangular grid** with the same
layout as the widely used `cylinder_wake.mat` (fields: X_star, U_star, p_star, t),
so it can be consumed by PINN/FNO pipelines that expect that format.

Geometry & defaults (can be changed via CLI):
  • Domain:  [x_min, x_max] × [y_min, y_max] = [1, 8] × [-2, 2]
  • Grid:    Sx × Sy = 100 × 50  (⇒ 5000 points)
  • Cylinder: center (xc, yc) = (2.0, 0.0), radius R = 0.5
  • Inlet:   parabolic u-inflow, v=0; Top/Bottom/Cylinder: no-slip; Outlet: ∂u/∂x=∂v/∂x=0, p=0
  • Re ≈ 100 with U≈1 and D=1  ⇒ ν = 0.01 (default), non-dimensional units
  • Time:    T_final = 19.9, dt_save = 0.1 (200 frames), internal sub-steps chosen by CFL

Numerics (robust & minimal):
  • Collocated grid; explicit advection (1st-order upwind); 5-point Laplacian diffusion
  • Brinkman penalization for the cylinder (no-slip):  du/dt += -beta * chi * (u - u_s), with u_s=0
  • Projection (Chorin) for incompressibility:
        u* = u^n + dt ( - (u·∇)u + νΔu - beta·chi·u )  + boundary enforcement
        Solve ∇²p^{n+1} = (1/dt) ∇·u*
        u^{n+1} = u* - dt ∇p^{n+1}
  • Pressure Poisson: weighted-Jacobi with mixed BC (p=0 at outlet, Neumann elsewhere)
  • Time step: dt = min(dt_adv, dt_diff, dt_penalty, dt_cap), with optional lid/inflow ramp

This is **not** an exact replica of the original Nektar++ spectral/hp DNS used in
Raissi/Karniadakis papers, but it reproduces the same *structure* (grid, time sampling,
BCs, Re) to generate a compatible dataset quickly.

Author: (your name)
License: MIT
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import scipy.io

# -----------------------------
# Utilities
# -----------------------------

def linspace_grid(xmin: float, xmax: float, Sx: int, ymin: float, ymax: float, Sy: int):
    x = np.linspace(xmin, xmax, Sx, dtype=np.float64)
    y = np.linspace(ymin, ymax, Sy, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing='ij')  # (Sx,Sy)
    dx = (xmax - xmin) / (Sx - 1)
    dy = (ymax - ymin) / (Sy - 1)
    return X, Y, x, y, dx, dy


def smooth_heaviside(phi: np.ndarray, eps: float) -> np.ndarray:
    """Smoothed indicator H(phi)≈1 if phi<0 (inside), 0 outside. eps ~ few*dx"""
    H = 0.5 * (1.0 - np.tanh(phi / (eps + 1e-12)))
    return H


@dataclass
class Cylinder:
    xc: float = 2.0
    yc: float = 0.0
    R: float = 0.5
    smooth_eps: float = 0.02  # smoothing length for penalization mask

    def mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        phi = np.sqrt((X - self.xc) ** 2 + (Y - self.yc) ** 2) - self.R
        return smooth_heaviside(phi, self.smooth_eps)


# -----------------------------
# Finite differences (collocated)
# -----------------------------

def ddx(f: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(f)
    out[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
    out[0, :] = (f[1, :] - f[0, :]) / dx
    out[-1, :] = (f[-1, :] - f[-2, :]) / dx
    return out


def ddy(f: np.ndarray, dy: float) -> np.ndarray:
    out = np.zeros_like(f)
    out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
    out[:, 0] = (f[:, 1] - f[:, 0]) / dy
    out[:, -1] = (f[:, -1] - f[:, -2]) / dy
    return out


def laplace(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = np.zeros_like(f)
    out[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dx * dx)
        + (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / (dy * dy)
    )
    # simple one-sided at boundaries (will be overwritten by BC enforcement)
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


def upwind_x(f: np.ndarray, u: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(f)
    # interior
    fm = f.copy()
    out[1:-1, :] = np.where(u[1:-1, :] >= 0.0,
                            (f[1:-1, :] - f[:-2, :]) / dx,
                            (f[2:, :] - f[1:-1, :]) / dx)
    # boundaries
    out[0, :] = (f[1, :] - f[0, :]) / dx
    out[-1, :] = (f[-1, :] - f[-2, :]) / dx
    return out


def upwind_y(f: np.ndarray, v: np.ndarray, dy: float) -> np.ndarray:
    out = np.zeros_like(f)
    out[:, 1:-1] = np.where(v[:, 1:-1] >= 0.0,
                            (f[:, 1:-1] - f[:, :-2]) / dy,
                            (f[:, 2:] - f[:, 1:-1]) / dy)
    out[:, 0] = (f[:, 1] - f[:, 0]) / dy
    out[:, -1] = (f[:, -1] - f[:, -2]) / dy
    return out


# -----------------------------
# Boundary conditions
# -----------------------------

def inflow_profile(y: np.ndarray, ymax: float, Umax: float = 1.0, kind: str = "parabolic") -> np.ndarray:
    if kind == "uniform":
        return Umax * np.ones_like(y)
    # Poiseuille-like between two walls at y=±ymax: u(y) = Umax * (1 - (y/ymax)^2)
    return Umax * (1.0 - (y / ymax) ** 2)


def apply_velocity_bcs(u: np.ndarray, v: np.ndarray,
                       X: np.ndarray, Y: np.ndarray,
                       x: np.ndarray, y: np.ndarray,
                       t: float, params) -> None:
    # Inlet (x = xmin): Dirichlet u_in(y), v=0
    u[0, :] = inflow_profile(y, ymax=abs(y[-1]), Umax=params.U_in, kind=params.inlet_kind)
    v[0, :] = 0.0
    # Top/Bottom walls: no-slip
    u[:, 0] = 0.0; v[:, 0] = 0.0
    u[:, -1] = 0.0; v[:, -1] = 0.0
    # Outlet (x = xmax): convective/zero-gradient (handled after advection-diffusion)


def apply_outlet_zero_gradient(u: np.ndarray, v: np.ndarray) -> None:
    u[-1, :] = u[-2, :]
    v[-1, :] = v[-2, :]


# -----------------------------
# Pressure Poisson solver: ∇²p = rhs with BCs (p=0 at outlet; Neumann elsewhere)
# -----------------------------

def pressure_poisson(rhs: np.ndarray, dx: float, dy: float,
                     outlet_dirichlet: bool = True,
                     iters: int = 400, omega: float = 0.8) -> np.ndarray:
    Sx, Sy = rhs.shape
    p = np.zeros_like(rhs)
    dx2, dy2 = dx * dx, dy * dy
    denom = 2.0 * (dx2 + dy2)
    for _ in range(iters):
        p_old = p
        # Jacobi sweep on interior
        p_new = p.copy()
        p_new[1:-1, 1:-1] = ((p[2:, 1:-1] + p[:-2, 1:-1]) * dy2 + (p[1:-1, 2:] + p[1:-1, :-2]) * dx2 - rhs[1:-1, 1:-1] * dx2 * dy2) / denom
        # Neumann at top/bottom: ∂p/∂y = 0 ⇒ copy neighbor
        p_new[:, 0] = p_new[:, 1]
        p_new[:, -1] = p_new[:, -2]
        # Neumann at inlet: ∂p/∂x = 0
        p_new[0, :] = p_new[1, :]
        # Outlet: Dirichlet p=0 (gauge)
        if outlet_dirichlet:
            p_new[-1, :] = 0.0
        p = omega * p_new + (1 - omega) * p
        # Optional: residual check could be added
    return p


# -----------------------------
# Time integration loop
# -----------------------------
@dataclass
class Params:
    Sx: int = 100
    Sy: int = 50
    x_min: float = 1.0
    x_max: float = 8.0
    y_min: float = -2.0
    y_max: float = 2.0
    nu: float = 1e-2            # viscosity
    U_in: float = 1.0           # inflow scale
    inlet_kind: str = "parabolic"  # or "uniform"
    beta_penalty: float = 200.0 # Brinkman penalization strength
    dt_cap: float = 5e-3        # max internal dt
    CFL: float = 0.5
    save_dt: float = 0.1
    T_final: float = 19.9
    ramp_time: float = 1.0      # inflow ramp duration
    cyl: Cylinder = Cylinder()


def simulate(params: Params) -> Dict[str, np.ndarray]:
    # Grid & arrays
    X, Y, x, y, dx, dy = linspace_grid(params.x_min, params.x_max, params.Sx,
                                       params.y_min, params.y_max, params.Sy)
    chi = params.cyl.mask(X, Y)  # 1 inside cylinder

    # State variables
    u = np.zeros((params.Sx, params.Sy), dtype=np.float64)
    v = np.zeros_like(u)
    p = np.zeros_like(u)

    # Storage for frames
    frames_t = np.arange(0.0, params.T_final + 1e-12, params.save_dt)
    T_rec = len(frames_t)
    U_rec = np.zeros((params.Sx, params.Sy, T_rec), dtype=np.float32)
    V_rec = np.zeros_like(U_rec)
    P_rec = np.zeros_like(U_rec)

    t = 0.0
    next_k = 0
    # initial ramp factor
    def inflow_scale(tt):
        if params.ramp_time <= 0: return 1.0
        a = min(1.0, max(0.0, tt / params.ramp_time))
        return 0.5 * (1 - math.cos(math.pi * a))

    while next_k < T_rec:
        # Enforce inflow/outflow/walls on current u,v
        apply_velocity_bcs(u, v, X, Y, x, y, t, params)

        # Compute time step
        umax = max(1e-8, float(np.max(np.abs(u))))
        vmax = max(1e-8, float(np.max(np.abs(v))))
        dt_adv = params.CFL * min(dx, dy) / max(umax, vmax)
        dt_diff = 0.25 * min(dx, dy) ** 2 / max(params.nu, 1e-12)
        dt_pen = 1.0 / (params.beta_penalty + 1e-12)
        dt = min(dt_adv, dt_diff, dt_pen, params.dt_cap)
        if t + dt > frames_t[next_k]:
            dt = frames_t[next_k] - t

        # Advection (upwind) & diffusion
        du_dx = upwind_x(u, u, dx); du_dy = upwind_y(u, v, dy)
        dv_dx = upwind_x(v, u, dx); dv_dy = upwind_y(v, v, dy)
        Lu = laplace(u, dx, dy)
        Lv = laplace(v, dx, dy)

        # Penalization inside cylinder: drives velocity → 0
        beta = params.beta_penalty
        # Predictor: u* = u + dt( -conv + nuΔu - beta χ u )
        u_star = u + dt * ( - (u * du_dx + v * du_dy) + params.nu * Lu - beta * chi * u )
        v_star = v + dt * ( - (u * dv_dx + v * dv_dy) + params.nu * Lv - beta * chi * v )

        # Re-enforce Dirichlet BCs on u* (inlet, walls)
        apply_velocity_bcs(u_star, v_star, X, Y, x, y, t + dt, params)
        apply_outlet_zero_gradient(u_star, v_star)

        # Pressure projection: ∇²p = (1/dt) ∇·u*
        div_u = ddx(u_star, dx) + ddy(v_star, dy)
        rhs = (1.0 / dt) * div_u
        p = pressure_poisson(rhs, dx, dy, outlet_dirichlet=True, iters=200, omega=0.8)

        # Correct velocities: u^{n+1} = u* - dt ∇p
        u = u_star - dt * ddx(p, dx)
        v = v_star - dt * ddy(p, dy)

        # Enforce BCs again & zero in cylinder (optional hard clamp)
        apply_velocity_bcs(u, v, X, Y, x, y, t + dt, params)
        u = (1.0 - chi) * u  # strong zeroing in solid
        v = (1.0 - chi) * v

        t += dt

        # Save a frame if we hit the sampling time
        if abs(t - frames_t[next_k]) <= 1e-12 or t > frames_t[next_k] - 1e-12:
            # Optional: lightweight smoothing at outlet to avoid spikes
            apply_outlet_zero_gradient(u, v)
            U_rec[:, :, next_k] = u.astype(np.float32)
            V_rec[:, :, next_k] = v.astype(np.float32)
            P_rec[:, :, next_k] = p.astype(np.float32)
            next_k += 1

    # Pack outputs in the .mat style used by Raissi et al.
    # X_star: (N,2) ordered row-major over the grid
    Sx, Sy = params.Sx, params.Sy
    X_star = np.column_stack([X.reshape(-1), Y.reshape(-1)])  # (N,2)
    U_star = np.stack([U_rec.reshape(Sx * Sy, -1), V_rec.reshape(Sx * Sy, -1)], axis=1)  # (N,2,T)
    p_star = P_rec.reshape(Sx * Sy, -1)  # (N,T)
    t_vec = frames_t.reshape(-1, 1)

    return {
        'X_star': X_star.astype(np.float64),
        'U_star': U_star.astype(np.float64),
        'p_star': p_star.astype(np.float64),
        't': t_vec.astype(np.float64),
        # extras
        'x': x, 'y': y,
    }


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description='Generate a cylinder wake dataset compatible with cylinder_wake.mat')
    p.add_argument('--Sx', type=int, default=100, help='Grid points in x (default 100)')
    p.add_argument('--Sy', type=int, default=50,  help='Grid points in y (default 50)')
    p.add_argument('--xmin', type=float, default=1.0)
    p.add_argument('--xmax', type=float, default=8.0)
    p.add_argument('--ymin', type=float, default=-2.0)
    p.add_argument('--ymax', type=float, default=2.0)
    p.add_argument('--nu', type=float, default=1e-2, help='Kinematic viscosity ν')
    p.add_argument('--U_in', type=float, default=1.0, help='Inflow scale')
    p.add_argument('--inlet', type=str, default='parabolic', choices=['parabolic','uniform'])
    p.add_argument('--save_dt', type=float, default=0.1, help='Sampling interval of saved frames')
    p.add_argument('--T', type=float, default=19.9, help='Final time')
    p.add_argument('--beta', type=float, default=200.0, help='Brinkman penalization strength (solid)')
    p.add_argument('--dt_cap', type=float, default=5e-3, help='Max internal dt')
    p.add_argument('--CFL', type=float, default=0.5, help='CFL for advection')
    p.add_argument('--ramp', type=float, default=1.0, help='Inflow ramp duration')
    p.add_argument('--xc', type=float, default=2.0)
    p.add_argument('--yc', type=float, default=0.0)
    p.add_argument('--R',  type=float, default=0.5)
    p.add_argument('--smooth', type=float, default=0.02, help='Smoothing length for cylinder mask')
    p.add_argument('--out', type=str, default='cylinder_wake_synthetic.mat')

    args = p.parse_args()

    params = Params(
        Sx=args.Sx, Sy=args.Sy,
        x_min=args.xmin, x_max=args.xmax, y_min=args.ymin, y_max=args.ymax,
        nu=args.nu, U_in=args.U_in, inlet_kind=args.inlet,
        beta_penalty=args.beta, dt_cap=args.dt_cap, CFL=args.CFL,
        save_dt=args.save_dt, T_final=args.T, ramp_time=args.ramp,
        cyl=Cylinder(xc=args.xc, yc=args.yc, R=args.R, smooth_eps=args.smooth)
    )

    data = simulate(params)

    scipy.io.savemat(args.out, mdict=data)
    print(f"Saved dataset to {args.out}\n"
          f"  X_star: {data['X_star'].shape}, U_star: {data['U_star'].shape}, p_star: {data['p_star'].shape}, t: {data['t'].shape}")


if __name__ == '__main__':
    main()
