# Visualize Poisson dataset on a regular grid: source f, analytic solution u*, and residual |Δψ + f|
import math
import numpy as np
import pandas as pd
import torch
from torch.autograd import grad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from utilities3 import *
from spline_models import interpolate_states
from setups import Dataset
from models import Net2d
from get_params import get_params

# ---- precision & seeds ----
torch.set_default_dtype(torch.float64)
torch.manual_seed(11)
np.random.seed(11)

device = torch.device("cuda")

# ---- helpers ----
def make_boundary_masks(B: int, H: int, W: int, pad: int = 1, device=None, dtype=torch.bool):
    """
    Restituisce:
      interior_mask: (B,1,H,W) True nell'interno (esclusa la cornice di spessore 'pad')
      boundary_mask: (B,1,H,W) True solo sulla cornice di spessore 'pad'
    """
    m = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
    # Pad for boundary
    m[:, :, :pad, :]  = True
    m[:, :, -pad:, :] = True
    m[:, :, :, :pad]  = True
    m[:, :, :, -pad:] = True
    boundary_mask = m.expand(B, -1, -1, -1)             # (B,1,H,W)
    interior_mask = (~m).expand(B, -1, -1, -1)          # (B,1,H,W)
    return interior_mask, boundary_mask

def masked_mse_mean(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B,H,W)   |   mask: (B,1,H,W) (True = tieni)
    """
    diff2 = (pred - target).unsqueeze(1).pow(2)          # (B,1,H,W)
    m = mask
    num = (diff2 * m).sum()
    den = m.sum().clamp_min(1)                           # evita div/0
    return num / den

# ----- PDE helpers -----
def f_source(xy: torch.Tensor) -> torch.Tensor:
    # xy: (N,2) in [0,1]^2
    x, y = xy[:, :1], xy[:, 1:]
    return torch.sin(2*math.pi*x) * torch.sin(2*math.pi*y)

def analytical_u(xy: torch.Tensor) -> torch.Tensor:
    # -Δu = f  =>  u*(x,y) = (1/(8π^2)) sin(2πx) sin(2πy)
    return f_source(xy) / (8*math.pi**2)

def laplacian_from_grid(u, gx, gy):

    du_dx, du_dy = grad(
        u, [gx, gy],
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )
    u_xx = grad(du_dx, gx, grad_outputs=torch.ones_like(du_dx),
                create_graph=True, retain_graph=True)[0]
    u_yy = grad(du_dy, gy, grad_outputs=torch.ones_like(du_dy),
                create_graph=True, retain_graph=True)[0]
    return u_xx + u_yy

# ----- Build regular grid -----
S = 128
x = torch.linspace(0.0, 1.0, S, device=device)
y = torch.linspace(0.0, 1.0, S, device=device)
X, Y = torch.meshgrid(x, y, indexing="ij")                   # (S,S)
XY = torch.stack([X, Y], dim=-1).reshape(-1, 2)              # (S*S,2)

# ---------- Params & tensors ----------
params          = get_params()
modes           = params.modes
width           = params.width
orders_v        = params.orders_v
epochs          = params.epochs
lbfgs_epochs    = params.lbfgs_epochs
learning_rate   = params.learning_rate
scheduler_step  = params.scheduler_step
scheduler_gamma = params.scheduler_gamma
pad = 5

gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1]).to(device)
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1]).to(device)

# Source and analytical solution
source = f_source(XY).reshape(1, S, S, 1)       # (1,S,S,1)
u_true = analytical_u(XY).reshape(1, S, S).to(device)     # (1,S,S)

zero = torch.zeros((1, S, S, 1), device=device)
B, H, W = u_true.shape
interior_mask, boundary_mask = make_boundary_masks(B, H, W, pad=pad, device=device)

# Input 3 channels: (x, y, f)
inp = torch.cat([ source, gridx, gridy], dim=-1)              # (1,S,S,3)

# ----- Model -----
model = Net2d(modes, width).cuda()


# ---------- Loss function ----------
def pinn_loss(lambda1, lambda2):
    out = model(inp)
    ψ, lap_ψ = interpolate_states(out, offset=torch.tensor([0.5, 0.5], device=device),
                                  orders_v=orders_v)  # (B,H,W)

    res = lap_ψ + source.squeeze(-1)  # Δψ + f
    loss_res = ((res**2) *  interior_mask.squeeze(1)).sum() / (interior_mask.squeeze(1)).sum()

    #Normalization for stabilisation of loss amplitude
    with torch.no_grad():
        denom = torch.sqrt((source.squeeze(-1)**2).mean()) + 1e-12
    res_norm = res / denom

    loss_res = masked_mse_mean(res_norm, torch.zeros_like(res_norm), interior_mask)
    loss_bc  = masked_mse_mean(ψ, u_true, boundary_mask)

    return lambda2*loss_res + lambda1* loss_bc, loss_res, loss_bc

# ---------- history plotting ----------
history = {'iter': [], 'phase': [], 'total': [], 'res': [], 'bc': []}

def log_point(i, phase, total, res, bc):
    history['iter'].append(int(i))
    history['phase'].append(phase)
    history['total'].append(float(total))
    history['res'].append(float(res))
    history['bc'].append(float(bc))



start_time = time.perf_counter()
adam_epochs = 2000  

# ======== Phases ========
phases = [
    {'name': 'full_loss_1', 'record': True,  'epochs': epochs, 'λ_sup': 1.0, 'λ_res':0.0},
    {'name': 'full_loss_2', 'record': True,  'epochs': epochs, 'λ_sup': 0.8, 'λ_res': 0.5},
    {'name': 'full_loss_3', 'record': True,  'epochs': epochs, 'λ_sup': 0.5, 'λ_res': 1.0}
]

# If you want to use the "no-reset" setting you should call the optimizer outside from the phase cycle
# and not inside the for cicle anymore

opt_adam = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(opt_adam, step_size=scheduler_step, gamma=scheduler_gamma)

for phase_idx, phase in enumerate(phases):

    print(f"\n=== Phase {phase['name']} | epochs {phase['epochs']} ===\n")
    # ---------- Adam ----------

    λ_sup    = phase['λ_sup'];    λ_res    = phase['λ_res']
    train_res_hist, train_bd_hist = [], []

    for ep in range(1, adam_epochs + 1):
        opt_adam.zero_grad(set_to_none=True)
        loss, loss_res, loss_bc = pinn_loss(λ_sup , λ_res)     
        loss.backward()
        opt_adam.step()
        
        log_point(ep, 'adam', loss.item(), loss_res.item(), loss_bc.item())

        if ep == 1 or ep % 50 == 0:
                
            print(f"Adam {ep:5d} | total {loss:.5e}  res {loss_res:.2e}  bc {loss_bc:.2e}")

elapsed = time.perf_counter() - start_time
print(f"\nAdam Training Time: {elapsed:.1f} seconds")

adam_end_iter = history['iter'][-1] if history['iter'] else 0

final_loss, lb_res, lb_bc = pinn_loss( λ_sup,  λ_res )
print(f"\nFinal loss {final_loss:.2e} res {lb_res:.2e}, bc {lb_bc:.2e})")

# ---------- Plot delle loss ----------
# Salva CSV semplice
with open("loss_history.csv", "w") as f:
    f.write("iter,phase,total,res,bc\n")
    for i, ph, t, r, b in zip(history['iter'], history['phase'], history['total'], history['res'], history['bc']):
        f.write(f"{i},{ph},{t:.10e},{r:.10e},{b:.10e}\n")

# Figura: curve log-scale e linea verticale per inizio LBFGS
fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
ax.semilogy(history['iter'], history['total'], label='total')
ax.semilogy(history['iter'], history['res'],   label='residual')
ax.semilogy(history['iter'], history['bc'],    label='boundary')

if adam_end_iter > 0:
    ax.axvline(adam_end_iter, linestyle='--', linewidth=1.0)
    ax.text(adam_end_iter, ax.get_ylim()[1]*0.8, 'LBFGS start', rotation=90,
            va='top', ha='right', fontsize=9)

ax.set_xlabel('iteration (Adam epochs → LBFGS steps)')
ax.set_ylabel('loss (log scale)')
ax.set_title('Training losses: Adam + L-BFGS')
ax.legend()
ax.grid(True, which='both', alpha=0.25)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("Saved: loss_curve.png e loss_history.csv")

# ---------- Evaluation & plots ----------
model.eval()
with torch.no_grad():
    out = model(inp)
    ψ, lap_ψ = interpolate_states(out, offset=torch.tensor([0.0, 0.0], device=device),
                                  orders_v=orders_v)
f_torch = source.squeeze(-1)
f_pred_torch = lap_ψ.squeeze(0)
u_true_img = u_true.squeeze(0).detach().cpu().numpy()        # (S,S)
f_true_img = source.squeeze(0).squeeze(-1).detach().cpu().numpy()        # (S,S)
u_pred_img = ψ.squeeze(0).detach().cpu().numpy()             # (S,S)
f_pred_img = lap_ψ.squeeze(0).detach().cpu().numpy()         # (S,S)

# Errore sui vincoli della PDE: vogliamo Δψ ≈ -f  =>  |Δψ + f|

err = np.abs(f_pred_img + f_true_img)
err_torch = f_torch + f_pred_torch

total_elapsed = time.perf_counter() - start_time
print(f"\nTotal wall-clock time: {total_elapsed:.1f} seconds")

fig, ax = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
for a, data, title in zip(ax, [u_true_img, u_pred_img, err[pad:-pad,pad:-pad]],
                          ["analytic ψ", "ψ PhIS-FNO", r"$|\Delta\psi+f|$"]):
    im = a.imshow(data, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    a.set_title(title)
    fig.colorbar(im, ax=a)

plt.savefig("poisson_eval.png", dpi=150, bbox_inches="tight")
print("Saved: poisson_eval.png")

# --- Interior residual statistics (NumPy) ---
# Use the 'pad' margin to exclude boundary points and evaluate only the interior domain
err_interior = err[pad:-pad, pad:-pad]          # 2D array (H-2*pad, W-2*pad)

# Compute mean and standard deviation of the absolute residual |Δψ + f|
# By default np.std uses population std (ddof=0). 
# If you prefer the sample std, use ddof=1.
mean_abs = float(np.mean(err_interior))
std_abs  = float(np.std(err_interior))          # or np.std(err_interior, ddof=1)

# (Optional) Root-mean-square error (L2 norm of the residual field)
rmse_abs = float(np.sqrt(np.mean(err_interior**2)))

print(f"Interior |Δψ+f|: mean ± std = {mean_abs:.3e} ± {std_abs:.3e}  (N = {err_interior.size})")

