#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_burgers1d_dataset.py
Crea un dataset 1D per l'equazione di Burgers viscosa su [0,1] periodico:
    u_t + 0.5 * d_x(u^2) = nu * u_xx
- Condizione iniziale: GRF periodico N(m, C) con
    C = sigma^2 (-Δ + tau^2 I)^(-gamma)
- Integrazione: pseudo-spettrale ETDRK4 (Cox–Matthews) con dealiasing 2/3
- Uscita: npy + mat contenenti:
    input : (Ns, Nx)     -> profili iniziali u(x, t=0)
    output: (Ns, steps, Nx)  -> snapshot a tempi t>0 equispaziati
    x     : (Nx,)
    t     : (steps,)     -> tempi salvati (escluso t=0)
    nu, gamma, tau, sigma

Esempio:
python make_burgers1d_dataset.py --Ns 100 --Nx 8192 --steps 200 --nu 1e-3 --gamma 2.5 --tau 7 --sigma 49 --dt 1e-4 --seed 0 --out data_burgers_1d
"""

import argparse, os, math
import numpy as np
from numpy.fft import rfft, irfft, fftfreq
from time import time
from scipy.io import savemat

# ----------------------------
# Utilità
# ----------------------------
def make_grid(Nx: int):
    """Griglia periodica [0,1), Nx punti."""
    x = np.linspace(0.0, 1.0, Nx, endpoint=False)
    return x

def dealias_mask_rfft(Nx: int):
    """
    Maschera 2/3-rule in spazio di Fourier reale (rfft).
    Per rfft(Nx): lunghezza spettro è Nx//2 + 1 con k=0..Nx/2.
    Annulla le alte frequenze oltre cutoff = floor((2/3)*(Nx/2)).
    """
    Kmax = Nx // 2
    cutoff = int(np.floor((2.0/3.0) * Kmax))
    m = np.ones(Kmax + 1, dtype=bool)
    m[(cutoff+1):] = False
    return m

def sample_grf_periodic(Nx: int, gamma: float, tau: float, sigma: float,
                        mean: float = 0.0, rng: np.random.Generator = None,
                        shift_half: bool = True):
    """
    Campiona un GRF periodico su [0,1) con spettro ~ sigma * ((2π k)^2 + tau^2)^(-gamma/2).
    Restituisce array reale u0 di lunghezza Nx.
    Implementazione via rFFT: imponiamo coniugio per real-valued e (opzionale) shift x->x-0.5
    applicando fattore di fase (-1)^k alle armoniche (equivalente alla traslazione di 0.5).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Frequenze rfft: k = 0..Nx/2
    k = np.arange(0, Nx//2 + 1, dtype=np.float64)
    two_pi = 2.0 * np.pi
    # Spettro target (deviazione std) per ciascun k>0 (k=0 gestito a parte)
    eigs = np.sqrt(2.0) * (abs(sigma) * ((two_pi*k)**2 + tau**2) ** (-gamma/2.0))
    # Nota: formula coerente con i .m (my_eigs); per k=0 non usiamo eigs, manteniamo mean

    # Coefficienti casuali complessi per k>0 con varianza controllata
    # rfft output: shape Nx//2+1, c[0] e c[-1] (Nyquist) devono essere reali.
    c = np.zeros(k.shape, dtype=np.complex128)

    # k=0 (media)
    c[0] = mean

    # k=1..K-1 (se esiste Nyquist), parte complessa
    Kmax = Nx//2
    if Kmax >= 1:
        # per k=1..Kmax-1
        if Kmax-1 >= 1:
            kk = np.arange(1, Kmax, dtype=int)
            std = eigs[kk]  # deviazione std per ampiezza
            re = rng.normal(0.0, std)
            im = rng.normal(0.0, std)
            ck = re + 1j*im
            if shift_half:
                ck *= ((-1.0) ** kk)  # fase per x->x-0.5
            c[kk] = ck

        # k=Kmax (Nyquist) reale (se Nx è pari)
        if Nx % 2 == 0:
            stdN = eigs[Kmax]
            reN = rng.normal(0.0, stdN)
            ckN = reN  # deve essere reale
            if shift_half:
                ckN *= ((-1.0) ** Kmax)
            c[Kmax] = ckN

    # u0 in spazio fisico: irfft restituisce reale di lunghezza Nx
    u0 = irfft(c, n=Nx)
    return u0

def burgers_etdrk4(u0: np.ndarray, nu: float, T: float, dt: float,
                   save_steps: int, dealias_mask: np.ndarray):
    """
    ETDRK4 pseudo-spettrale per Burgers 1D periodico su [0,1].
    - u0: (Nx,) stato iniziale
    - nu: viscosità
    - T: tempo finale (tip: 1.0)
    - dt: passo temporale
    - save_steps: quante istantanee equispaziate (escludendo t=0) salvare in (0,T]
    - dealias_mask: maschera booleana rfft per 2/3-rule
    Ritorna:
      t_save: (save_steps,) tempi salvati
      Usave: (save_steps, Nx)
    """
    Nx = u0.size
    x = make_grid(Nx)
    # Frequenze per derivate: con rfft, k=0..Nx/2
    k = np.arange(0, Nx//2 + 1, dtype=np.float64)
    two_pi = 2.0 * np.pi
    ik = 1j * (two_pi * k)  # per derivata spaziale
    L = - (two_pi*k)**2     # laplaciano in spettro (∂xx -> -(2πk)^2)
    Lu = nu * L

    # Coefficienti ETDRK4 (Cox–Matthews) per operator lineare Lu
    E = np.exp(Lu * dt)
    E2 = np.exp(Lu * dt / 2.0)
    # Coefficienti phi via serie tratte dal paper Kassam–Trefethen (stabile)
    # Usando media a trapezi su M punti (default M=32)
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)  # punti su semicirconferenza
    L_mat = Lu[:, None]  # (K+1,1)
    Q = dt * np.mean((np.exp(r*dt) - 1.0) / (r*dt) , axis=0)  # scalare (media, ma non dipende da L)
    # Per vettori L diversi, calcoliamo le phi come medie su r:
    def phi1(z): return (np.exp(z) - 1.0) / z
    def phi2(z): return (np.exp(z) - 1.0 - z) / (z*z)
    def phi3(z): return (np.exp(z) - 1.0 - z - 0.5*z*z) / (z*z*z)
    # Evitiamo divisioni 0/0 gestendo z=0 con limiti:
    def safe_phi(func, z):
        out = np.empty_like(z, dtype=np.complex128)
        mask0 = (z == 0)
        out[~mask0] = func(z[~mask0])
        # limiti:
        # phi1(0)=1, phi2(0)=1/2, phi3(0)=1/6
        if func == phi1:
            out[mask0] = 1.0
        elif func == phi2:
            out[mask0] = 0.5
        else:
            out[mask0] = (1.0/6.0)
        return out

    phi1L = safe_phi(phi1, Lu * dt)
    phi2L = safe_phi(phi2, Lu * dt)
    phi3L = safe_phi(phi3, Lu * dt)

    # Preparazione salvataggi
    nsteps = int(np.round(T / dt))
    if nsteps <= 0:
        raise ValueError("T/dt deve essere >= 1 passo.")
    # indici di salvataggio equispaziati (escludi t=0)
    save_idx = np.linspace(1, nsteps, save_steps, dtype=int)
    t_save = save_idx * dt
    Usave = np.empty((save_steps, Nx), dtype=np.float64)

    # Spettro iniziale
    u_hat = rfft(u0)
    # Funzione per calcolare termine nonlineare N(u) = -0.5 ∂x (u^2)
    def Nl(u_hat_local):
        # dealias
        u_hat_dealiased = u_hat_local.copy()
        u_hat_dealiased[~dealias_mask] = 0.0
        u = irfft(u_hat_dealiased, n=Nx)
        uu = u * u
        uu_hat = rfft(uu)
        # derivata spaziale di uu: d/dx(u^2) -> ik * fft(u^2)
        d_uu_hat = ik * uu_hat
        return -0.5 * d_uu_hat

    # Time stepping ETDRK4
    s_ptr = 0
    for n in range(1, nsteps+1):
        # ETDRK4 stages in spettro
        Nv  = Nl(u_hat)
        a   = E2 * u_hat + dt * 0.5 * E2 * Nv
        Na  = Nl(a)
        b   = E2 * u_hat + dt * 0.5 * E2 * Na
        Nb  = Nl(b)
        c   = E  * u_hat + dt * E * Nb
        Nc  = Nl(c)

        u_hat = E * u_hat + dt * (phi1L * Nv + 2.0 * (phi2L * (Na + Nb)) + phi3L * Nc)

        if n in save_idx:
            # Salva stato *dealiased* per coerenza
            u_hat_out = u_hat.copy()
            u_hat_out[~dealias_mask] = 0.0
            Usave[s_ptr] = irfft(u_hat_out, n=Nx).real
            s_ptr += 1

    return t_save, Usave

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Genera dataset Burgers 1D (GRF + ETDRK4).")
    p.add_argument("--Ns", type=int, default=100, help="Numero di campioni (trajectories).")
    p.add_argument("--Nx", type=int, default=2**13, help="Punti griglia spaziale (consigliato pari).")
    p.add_argument("--steps", type=int, default=2, help="Numero snapshot salvati in (0,1].")
    p.add_argument("--nu", type=float, default=1e-1, help="Viscosità (nu).")
    p.add_argument("--gamma", type=float, default=2.5, help="Parametro GRF gamma.")
    p.add_argument("--tau", type=float, default=7.0, help="Parametro GRF tau.")
    p.add_argument("--sigma", type=float, default=49.0, help="Parametro GRF sigma.")
    p.add_argument("--mean", type=float, default=0.0, help="Media m del GRF.")
    p.add_argument("--dt", type=float, default=1e-4, help="Passo temporale per integrazione.")
    p.add_argument("--T", type=float, default=0.1, help="Tempo finale.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--out", type=str, default="burgers1d_dataset", help="Prefisso output (senza estensione).")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    Nx = int(args.Nx)
    Ns = int(args.Ns)
    steps = int(args.steps)
    dt = float(args.dt)
    T = float(args.T)

    if Nx % 2 != 0:
        raise ValueError("Nx deve essere pari per rfft/ETDRK4 più pulito (Nyquist).")

    x = make_grid(Nx)
    t_save = np.linspace(dt, T, steps)  # solo informativo; reale da integratore
    mask = dealias_mask_rfft(Nx)

    input_ic = np.empty((Ns, Nx), dtype=np.float64)
    output_ts = np.empty((Ns, steps, Nx), dtype=np.float64)

    t0 = time()
    for j in range(Ns):
        # 1) IC ~ GRF periodico (match ai tuoi .m: gamma,tau,sigma)
        u0 = sample_grf_periodic(
            Nx=Nx,
            gamma=args.gamma,
            tau=args.tau,
            sigma=args.sigma,
            mean=args.mean,
            rng=rng,
            shift_half=True  # come u(t-0.5) nei .m
        )
        u0 *= 1/ (np.max(np.abs(u0)) + 1e-12)
        input_ic[j] = u0

        # 2) Integrazione Burgers con ETDRK4
        t_snap, Utraj = burgers_etdrk4(
            u0=u0, nu=args.nu, T=T, dt=dt, save_steps=steps, dealias_mask=mask
        )
        output_ts[j] = Utraj

        if (j+1) % max(1, Ns//10) == 0 or j == Ns-1:
            print(f"[{j+1}/{Ns}]")

    elapsed = time() - t0
    print(f"Done. {Ns} samples in {elapsed:.1f}s  (~{elapsed/max(1,Ns):.2f}s/sample)")

    # Salvataggio
    base = args.out

    # .mat per compatibilità con pipeline MATLAB/FNO classiche
    mat_dict = {
        "a": input_ic,
        "u": output_ts,
        "x": x,
        "t": t_snap,
        "nu": float(args.nu),
        "gamma": float(args.gamma),
        "tau": float(args.tau),
        "sigma": float(args.sigma),
        "dt": float(dt),
        "T": float(T),
        "seed": int(args.seed),
        "Nx": int(Nx),
        "Ns": int(Ns),
        "steps": int(steps),
    }
    savemat(base + ".mat", mat_dict)
    print(f"Saved: {base}.mat")

if __name__ == "__main__":
    main()
