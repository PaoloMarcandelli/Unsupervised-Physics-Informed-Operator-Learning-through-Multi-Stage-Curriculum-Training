# File: ns_2d.py
import torch
import math
from random_fields import GaussianRF
from timeit import default_timer
import scipy.io

# Updated Navier–Stokes 2D solver using torch.fft API
# w0: initial vorticity
# f: forcing term
# visc: viscosity (1/Re)
# T: final time
# delta_t: internal time-step
# record_steps: number of in-time snapshots to record
def make_dipole(s, device, x0=0.35, y0=0.5, sep=0.25, sigma=0.06, A=8.0):
    x = torch.linspace(0, 1, s+1, device=device)[:-1]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    # centri dei due vortici
    x1, y1 = x0 - sep/2, y0
    x2, y2 = x0 + sep/2, y0
    r1 = ( (X-x1)**2 + (Y-y1)**2 ) / (2*sigma**2)
    r2 = ( (X-x2)**2 + (Y-y2)**2 ) / (2*sigma**2)
    w = A*(torch.exp(-r1) - torch.exp(-r2))
    return w
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    N = w0.size(-1)
    k_max = N // 2
    steps = math.ceil(T / delta_t)
    record_time = max(1, steps // record_steps)

    # Initial Fourier space fields
    w_h = torch.fft.fftn(w0, dim=(-2, -1))
    f_h = torch.fft.fftn(f, dim=(-2, -1))
    if f_h.ndim < w_h.ndim:
        f_h = f_h.unsqueeze(0)

    # Wavenumber grids
    ky = torch.cat((torch.arange(0, k_max, device=w0.device),
                    torch.arange(-k_max, 0, device=w0.device)))
    k_y = ky.view(1, -1).repeat(N, 1)
    k_x = k_y.t()

    lap = 4 * (math.pi ** 2) * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0
    mask = ((k_x.abs() <= (2/3)*k_max) & (k_y.abs() <= (2/3)*k_max)).float()
    dealias = mask.unsqueeze(0)

    # Prepare storage
    batch_shape = w0.shape[:-2]
    sol = torch.zeros(*batch_shape, N, N, record_steps, device=w0.device)       # vorticità
    sol_ux = torch.zeros(*batch_shape, N, N, record_steps, device=w0.device)    # velocità x
    sol_uy = torch.zeros(*batch_shape, N, N, record_steps, device=w0.device)    # velocità y
    sol_t = torch.zeros(record_steps, device=w0.device)

    t = 0.0
    c = 0
    print(f"  -> Starting time-stepping: {steps} steps, recording every {record_time} steps...")
    for j in range(steps):
        # Stream function in Fourier
        psi_h = w_h / lap
        psi_h[..., 0, 0] = 0.0  # media nulla per ψ

        # Velocity in Fourier (u = ∂ψ/∂y, v = -∂ψ/∂x)
        u_hat = 1j * 2 * math.pi * k_y * psi_h
        v_hat = -1j * 2 * math.pi * k_x * psi_h
        u = torch.fft.ifftn(u_hat, dim=(-2, -1)).real   # u_x
        v = torch.fft.ifftn(v_hat, dim=(-2, -1)).real   # u_y

        # Vorticity gradients
        w_x = torch.fft.ifftn(1j * 2 * math.pi * k_x * w_h, dim=(-2, -1)).real
        w_y = torch.fft.ifftn(1j * 2 * math.pi * k_y * w_h, dim=(-2, -1)).real

        # Nonlinear term
        nonlinear = u * w_x + v * w_y
        F_h = torch.fft.fftn(nonlinear, dim=(-2, -1)) * dealias

        # Time integration (Crank–Nicolson)
        num = -delta_t * F_h + delta_t * f_h + (1 - 0.5 * delta_t * visc * lap) * w_h
        den = 1 + 0.5 * delta_t * visc * lap
        w_h = num / den

        t += delta_t
        # Record snapshots
        if (j + 1) % record_time == 0:
            idx = c
            w = torch.fft.ifftn(w_h, dim=(-2, -1)).real
            sol[..., idx] = w
            sol_ux[..., idx] = u
            sol_uy[..., idx] = v
            sol_t[idx] = t
            print(f"    Timestep {j+1}/{steps}: recorded snapshot {idx+1}/{record_steps} at t={t:.4f}")
            c += 1
    print(f"  -> Completed time-stepping: recorded {c} snapshots.")
    return sol, sol_t, sol_ux, sol_uy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    s = 256
    N = 20
    print(f"Dataset settings -> samples: {N}, grid size: {s}x{s}")
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    x = torch.linspace(0, 1, s+1, device=device)[:-1]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    #f = 0.1 * (torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

    f = - 2 * torch.cos(2*math.pi*2*(Y))
    

    record_steps = 60
    a = torch.zeros(N, s, s, device=device)                            # vorticità iniziale
    u = torch.zeros(N, s, s, record_steps, device=device)              # vorticità nel tempo
    ux = torch.zeros(N, s, s, record_steps, device=device)             # velocità x nel tempo
    uy = torch.zeros(N, s, s, record_steps, device=device)             # velocità y nel tempo

    bsize = 20
    c = 0
    t0 = default_timer()
    print("Starting dataset generation...")
    for j in range(N // bsize):
        print(f"Sampling initial vorticity for batch {j+1}/{N//bsize}...")
        w0 = GRF.sample(bsize).to(device)
        print(f"Solving Navier–Stokes for batch {j+1}...")
        sol, sol_t, sol_ux, sol_uy = navier_stokes_2d(
            w0, f, visc=2e-3, T=15.0, delta_t=1e-4, record_steps=record_steps
        )
        a[c:c+bsize]  = w0
        u[c:c+bsize]  = sol
        ux[c:c+bsize] = sol_ux
        uy[c:c+bsize] = sol_uy
        c += bsize
        elapsed = default_timer() - t0
        print(f"Completed batch {j+1}: total samples {c}, elapsed time {elapsed:.2f}s")

    print("Saving dataset to 'ns_data.mat'...")
    scipy.io.savemat(
        'ns_data_Kolmogorov.mat',
        mdict={
            'a': a.cpu().numpy(),          # w0
            'u': u.cpu().numpy(),          # w(t)
            't': sol_t.cpu().numpy()       # tempi
        }
    )
    print("Dataset saved successfully.")

if __name__ == '__main__':
    main()
