import os, random, numpy as np
SEED = 10
#SEED = 0 # Kolmogorov

# Deve essere prima che PyTorch inizializzi CUDA/cublas
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # per determinismo cuBLAS (Ampere+)

random.seed(SEED)
np.random.seed(SEED)

import math
from timeit import default_timer

import numpy as np
import pandas as pd

import torch  # <--- dopo gli env var

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Pytorch deterministico
torch.use_deterministic_algorithms(True)   # se qualche op non lo è, solleva eccezione
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# niente TF32/AMP (per riproducibilità piena)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
USE_AMP = False

import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utilities3 import *
from spline_models import interpolate_states
from setups import Dataset
from models import Net2d, fluid_model
from get_params import get_params


def toCuda(x):
    if isinstance(x, (tuple, list)):
        return [toCuda(xi) for xi in x]
    return x.cuda(non_blocking=True)

def spectral_ops(omega, dealias=None):
    B, S, _ = omega.shape
    kmax = S//2
    ky = torch.cat([torch.arange(0, kmax, device=omega.device),
                    torch.arange(-kmax, 0, device=omega.device)])
    kx = ky.view(1, -1).repeat(S, 1).t()
    ky = ky.view(1, -1).repeat(S, 1)

    lap = 4*math.pi**2*(kx**2 + ky**2)
    lap[0, 0] = 1.0

    w_h   = torch.fft.fftn(omega, dim=(-2, -1))
    psi_h = w_h / lap
    psi_h[..., 0, 0] = 0.0              # (2)

    u_hat = 1j*2*math.pi*ky * psi_h     # u = ∂ψ/∂y
    v_hat = -1j*2*math.pi*kx * psi_h    # v = -∂ψ/∂x
    u = torch.fft.ifftn(u_hat, dim=(-2, -1)).real
    v = torch.fft.ifftn(v_hat, dim=(-2, -1)).real

    wx = torch.fft.ifftn(1j*2*math.pi*kx * w_h, dim=(-2, -1)).real
    wy = torch.fft.ifftn(1j*2*math.pi*ky * w_h, dim=(-2, -1)).real
    lapw = torch.fft.ifftn(-lap * w_h, dim=(-2, -1)).real  # Δω in spazio fisico

    # convettivo de-aliased, coerente col solver (1)
    conv = u * wx + v * wy
    if dealias is not None:
        conv_h = torch.fft.fftn(conv, dim=(-2,-1))
        conv_h = conv_h * dealias
        conv   = torch.fft.ifftn(conv_h, dim=(-2,-1)).real

    return u, v, wx, wy, lapw, conv

# costruisci una sola volta fuori dal loop
def make_dealias_mask(S, device):
    kmax = S//2
    ky = torch.cat([torch.arange(0,kmax,device=device),
                    torch.arange(-kmax,0,device=device)])
    k_y = ky.view(1,-1).repeat(S,1)
    k_x = k_y.t()
    mask = ((k_x.abs() <= (2/3)*kmax) & (k_y.abs() <= (2/3)*kmax)).float()
    return mask


def lp_on_mask(myloss: LpLoss, x: torch.Tensor, y: torch.Tensor, mask_bool: torch.Tensor):
    """
    x, y: (B,S,S,1) OR (B,S,S)
    mask_bool: (S*S,) or (B,S,S) [bool]
    Returns a *mean* over selected entries consistent with LpLoss(size_average=True).
    """
    if x.dim() == 4:  # (B,S,S,1) -> (B,S*S)
        X = x[..., 0].reshape(x.size(0), -1)
        Y = y[..., 0].reshape(y.size(0), -1)
    else:             # (B,S,S)
        X = x.reshape(x.size(0), -1)
        Y = y.reshape(y.size(0), -1)

    if mask_bool.dim() == 1:  # shared mask (S*S,)
        return myloss(X[:, mask_bool], Y[:, mask_bool])

    # per-sample mask (B,S,S)
    loss = 0.0
    for i in range(X.size(0)):
        mb = mask_bool[i].reshape(-1)
        loss = loss + myloss(X[i:i+1, mb], Y[i:i+1, mb])
    return loss / X.size(0)


# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':

    params = get_params()

    # TRAIN_PATH = 'ns_data_1e-4.mat'
    # TEST_PATH  = 'ns_data_1e-4.mat'

    TRAIN_PATH = 'ns_data_1e-3.mat'
    TEST_PATH  = 'ns_data_1e-3.mat'

    # TRAIN_PATH = 'ns_data_Kolmogorov.mat'
    # TEST_PATH  = 'ns_data_Kolmogorov.mat'

    # Hyper-params
    ntrain      = params.ntrain
    ntest       = params.ntest
    modes       = params.modes
    width       = params.width
    batch_size  = params.bsize

    epochs          = params.epochs
    lbfgs_epochs    = params.lbfgs_epochs
    learning_rate   = params.learning_rate
    scheduler_step  = params.scheduler_step
    scheduler_gamma = params.scheduler_gamma

    sub         = params.sub
    S           = params.S
    T_in        = params.T_in
    T           = params.T
    step        = params.step
    nu          = params.nu
    rf          = params.rf
    orders_v    = params.orders_v
    n_samples   = params.n_samples

    offset = 40 # NS nu=1e-3
    #offset = 20 # NS nu=1e-4
    #offset = 30 #Kolmogorov

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    
    path = f'UNet_ns_fourier{ntrain}_Adam_ep{epochs}_visc{nu}_m{modes}_w{width}_T{T}'
    #path = f'UNet_kolmogorov_{ntrain}_Adam_ep{epochs}_visc{nu}_m{modes}_w{width}_T{T}'
    path_model     = os.path.join('model',  path)
    path_image     = os.path.join('image',  path)
    path_loss_dir  = os.path.join('loss',   path)
    os.makedirs(path_image, exist_ok=True)
    os.makedirs(path_loss_dir, exist_ok=True)
    os.makedirs(os.path.dirname(path_model), exist_ok=True)

    # Model


    #Modello UNET
    
    model = fluid_model(params.orders_v).to(device)

    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('u')[:ntrain, ::sub, ::sub, offset: offset+T_in]
    train_u = reader.read_field('u')[:ntrain, ::sub, ::sub,  offset+T_in: offset+T+T_in]

    reader = MatReader(TEST_PATH)
    test_a  = reader.read_field('u')[-ntest:, ::sub, ::sub,  offset: offset+T_in]
    test_u  = reader.read_field('u')[-ntest:, ::sub, ::sub,  offset+T_in: offset+T+T_in]

    # Shapes & grids
    train_a = train_a.reshape(ntrain, S, S, T_in)
    test_a  = test_a.reshape(ntest,  S, S, T_in)

    gridx = torch.linspace(0, 1, S, device=device).view(1, S, 1, 1).repeat(1, 1, S, 1)
    gridy = torch.linspace(0, 1, S, device=device).view(1, 1, S, 1).repeat(1, S, 1, 1)

    train_a = torch.cat([
        gridx.repeat(ntrain, 1, 1, 1),
        gridy.repeat(ntrain, 1, 1, 1),
        train_a.to(device)
    ], dim=-1)

    test_a = torch.cat([
        gridx.repeat(ntest, 1, 1, 1),
        gridy.repeat(ntest, 1, 1, 1),
        test_a.to(device)
    ], dim=-1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u.to(device)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u.to(device)),
        batch_size=batch_size, shuffle=False
    )

    # Losses
    loss_data = LpLossSafe(size_average=True)   # metrics & full-field
    loss_bd   = LpLoss(size_average=True)   # boundary masked loss
    mse       = nn.MSELoss(reduction='mean')  # residual, smoothness, etc.

    # Hidden field dataset (for spline decoder)
    v_size  = np.prod([i+1 for i in orders_v])


 

############################################################################
######################## Navier Stokes Flow ##################################
############################################################################

    pad     = 1 # Navier Stokes
    dataset = Dataset(batch_size, v_size, S, rf, n_samples, pad)
    x = torch.linspace(0, 1, S+1, device=device)[:-1]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    f = 0.1 * (torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y))) #Navier Stokes Dynamics

    f0 = f.unsqueeze(0)  # (1,S,S)

    # Training phases for Navier Stokes
    phases = [
        {'name': 'boundary_only_1', 'record': False, 'epochs': epochs, 'λ_sup': 1.0, 'λ_res': 0.0, 'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2},
        {'name': 'boundary_only_2', 'record': False, 'epochs': epochs, 'λ_sup': 1.0, 'λ_res': 0.0, 'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2},
        {'name': 'full_loss_1',     'record': True,  'epochs': epochs, 'λ_sup': 0.8, 'λ_res': 0.5, 'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2},
        {'name': 'full_loss_2',     'record': True,  'epochs': epochs, 'λ_sup': 0.5, 'λ_res': 1.0, 'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2},
        {'name': 'full_loss_3',     'record': True,  'epochs': epochs, 'λ_sup': 0.2, 'λ_res': 1.5, 'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2}
    ]


############################################################################
######################## Kolmogorov Flow ##################################
############################################################################

    # pad     = 5 # Kolmogorov
    # dataset = Dataset(batch_size, v_size, S, rf, n_samples, pad)

    # # Forcing (broadcasted later on batch)
    # n = 2            # numero d’onda della forzante (classico: 1,2,4,...)

    # x = torch.linspace(0, 1, S+1, device=device)[:-1]
    # X, Y = torch.meshgrid(x, x, indexing='ij')
    # f =  - 2 * torch.cos(2*math.pi*n*(Y)) #Kolmogorov
    # f0 = f.unsqueeze(0)  # (1,S,S)


    # # Training phases for Kolmogorov Flow
    # phases = [
    #     {'name': 'boundary_only_1', 'record': False, 'epochs': epochs, 'λ_sup': 1.0, 'λ_res': 0.2, 'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2},
    #     {'name': 'full_loss_1',     'record': True,  'epochs': epochs, 'λ_sup': 0.8, 'λ_res': 0.5, 'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2},
    #     {'name': 'full_loss_2',     'record': True,  'epochs': epochs, 'λ_sup': 0.5, 'λ_res': 1.0, 'λ_smooth': 1e-1, 'λ_tv': 1e-2, 'λ_mean': 1e-2}
    # ]


    dt = 0.25



    n_steps = T // step
    dealias = make_dealias_mask(S, device)  # (S,S)
    dealias = dealias.unsqueeze(0)          # broadcast (1,S,S)
    for phase in phases:
        print(f"\n=== Phase {phase['name']} | epochs {phase['epochs']} ===\n")

        λ_sup    = phase['λ_sup']
        λ_res    = phase['λ_res']
        λ_smooth = phase['λ_smooth']
        λ_tv     = phase['λ_tv']
        λ_mean   = phase['λ_mean']

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        # Accumulator for CSV
        train_res_hist, train_bd_hist, train_full_hist = [], [], []
        test_step_hist,  test_full_hist  = [], []

        for ep in range(phase['epochs']):
            model.train()
            t_start = default_timer()

            # Per-epoch accumulators (means computed later)
            ep_res_sum = 0.0
            ep_bd_sum  = 0.0
            ep_step_sum = 0.0
            ep_full_sum = 0.0

            for xx, yy in train_loader:
                xx, yy = toCuda([xx, yy])
                optimizer.zero_grad()

                v_mask = toCuda(dataset.ask())  # v_mask: (B,S,S)
                interior_mask = (1 - v_mask).unsqueeze(1)     # (B,1,S,S)
                bm_bool_shared = (v_mask[0] > 0).reshape(-1)  # (S*S,)

                loss_total = 0.0

                for t in range(0, T, step):
                    y = yy[..., t:t+step]     # (B,S,S,1)

                    im = model(xx) #UNET


                    ω, v, grad_ω, lap_ω = interpolate_states(
                        im, offset=torch.tensor([0.0, 0.0], device=device),
                        orders_v=orders_v
                    ) #(B,S,S,9)

                    dω_dt = (ω[...,0] - xx[..., -1])/dt
                    # Residual: dω/dt + v·∇ω - νΔω - f = 0
                    # conv  = v[..., 0]*grad_ω[..., 0] + v[..., 1]*grad_ω[..., 1]
                    # lap_s = lap_ω[..., 0]
                    w_now = ω[..., 0]          # (B,S,S)
                    u, v, wx, wy, lapw, conv = spectral_ops(w_now, dealias)

                    lap_s = lapw               # (B,S,S)
                    f_batch = f0.expand(xx.size(0), -1, -1)
                    # ns_res = (dω_dt + conv - nu*lap_s - f_batch)  # (B,S,S)
                    # ns_res = torch.nan_to_num(ns_res, nan=0.0)

                    # Residual loss (MSE only on interior)
                    # res = ns_res.unsqueeze(1)                   # (B,1,S,S)
                    # num_interior = interior_mask.sum().clamp_min(1)
                    # loss_res_step = ((res**2 * interior_mask).sum() / num_interior)
                    # scale per–term (RMS)

                    # residuo normalizzato, pesi comparabili a ν piccolo
                    r = dω_dt + conv - nu*lap_s - f_batch

                    # MSE SOLO su interior
                    res = r.unsqueeze(1)                                    # (B,1,S,S)
                    wgt = interior_mask                                     # (B,1,S,S)
                    loss_res_step = ( (res**2 * wgt).sum() / wgt.sum().clamp_min(1) )

                    # Boundary supervised loss (masked Lp)
                    loss_sup_step = lp_on_mask(loss_bd, ω, y, bm_bool_shared)

                    # Smoothness (L2 grad) & TV (L1 grad) on ω[...,0]
                    w = ω[..., 0]  # (B,H,W)
                    dx = w[:, 1:, :] - w[:, :-1, :]
                    dy = w[:, :, 1:] - w[:, :, :-1]
                    smooth_loss = (dx.pow(2).mean() + dy.pow(2).mean())
                    tv_loss = (dx.abs().mean() + dy.abs().mean())

                    # Zero-mean gauge on v
                    v_mean_x = v[..., 0].mean(dim=(-2, -1))
                    v_mean_y = v[..., 1].mean(dim=(-2, -1))
                    mean_v_loss = (v_mean_x.pow(2) + v_mean_y.pow(2)).mean()

                    loss_step = (
                        λ_sup   * loss_sup_step +
                        λ_res   * loss_res_step +
                        λ_smooth* smooth_loss   +
                        λ_tv    * tv_loss       +
                        λ_mean  * mean_v_loss
                    )

                    loss_total = loss_total + loss_step

                    # Logging accumulators (sums over steps & batches)
                    ep_res_sum += loss_res_step.item()
                    ep_bd_sum  += loss_sup_step.item()
                    ep_step_sum += loss_step.item()

                    # roll the autoregressive input
                    if t == 0:
                        pred = ω
                    else:
                        pred = torch.cat([pred, ω], dim=-1)

                    xx = torch.cat([
                        gridx.repeat(batch_size, 1, 1, 1),
                        gridy.repeat(batch_size, 1, 1, 1),
                        xx[..., 2+step:], ω

                    ], dim=-1)

                loss_total = loss_total 
                loss_total.backward()
                optimizer.step()

                # Full-field metric (after sequence predicted)
                ep_full_sum += loss_data(
                    pred.reshape(batch_size, -1),
                    yy.reshape(batch_size, -1)
                ).item()

            # ---- End epoch: normalize logs ----
            num_train_batches = len(train_loader)
            denom = num_train_batches * n_steps
            train_res_mean  = ep_res_sum  / max(1, denom)
            train_bd_mean   = ep_bd_sum   / max(1, denom)
            train_step_mean = ep_step_sum / max(1, denom)
            train_full_mean = ep_full_sum / max(1, num_train_batches)

            # ------ Eval on test ------
            model.eval()
            test_step_sum = 0.0
            test_full_sum = 0.0
            with torch.no_grad():
                for xx, yy in test_loader:
                    xx, yy = toCuda([xx, yy])
                    v_mask = toCuda(dataset.ask())
                    for t in range(0, T, step):
                        y = yy[..., t:t+step]
                        im = model(xx)
                        omega, v, grad_omega, lap_omega = interpolate_states(
                            im,
                            offset=torch.tensor([0.0, 0.0], device=device), orders_v=orders_v
                        )

                        # metric on this step (unmasked full-field Lp)
                        test_step_sum += loss_data(
                            omega.reshape(batch_size, -1), y.reshape(batch_size, -1)
                        ).item()

                        B = xx.shape[0]  # batch reale

                        # === DEBUG OUTLIER CHECK QUI ===
                        if torch.isfinite(omega).all():
                            num = torch.linalg.vector_norm((omega - y).reshape(B,-1), 2, dim=1).mean().item()
                            den = torch.linalg.vector_norm(y.reshape(B,-1), 2, dim=1).mean().item()
                            mx  = omega.abs().max().item()
                            if num / max(den, 1e-6) > 1e3:
                                print(f"[DBG] step outlier: num={num:.3e} den={den:.3e} max|ω|={mx:.3e}")

                        if t == 0:
                            pred_t = omega
                        else:
                            pred_t = torch.cat((pred_t, omega), -1)

                        xx = torch.cat([
                            gridx.repeat(batch_size, 1, 1, 1),
                            gridy.repeat(batch_size, 1, 1, 1),
                            xx[..., 2+step:], omega
                        ], dim=-1)
 

                    test_full_sum += loss_data(
                        pred_t.reshape(batch_size, -1), yy.reshape(batch_size, -1)
                    ).item()

            test_step_mean = test_step_sum / max(1, len(test_loader)*n_steps)
            test_full_mean = test_full_sum / max(1, len(test_loader))

            elapsed = default_timer() - t_start
            scheduler.step()

            # Console print (means only)
            print(
                f"Epoch {ep+1:3d}/{phase['epochs']} │ Time: {elapsed:6.2f}s │ "
                f"Train res step: {train_res_mean:8.4e} │ "
                f"Train bd step:  {train_bd_mean:8.4e} │ "
                f"Total step:     {train_step_mean:8.4e} │ "
                f"Train full:     {train_full_mean:8.4e} │ "
                f"Test step:      {test_step_mean:8.4e} │ "
                f"Test full:      {test_full_mean:8.4e}"
            )

            # Save per-epoch history
            train_bd_hist.append(train_bd_mean)
            train_full_hist.append(train_full_mean)
            test_step_hist.append(test_step_mean)
            test_full_hist.append(test_full_mean)
            if phase['record']:
                train_res_hist.append(train_res_mean)

        # Phase CSV
        idx = list(range(1, phase['epochs']+1))
        data = {
            'train_bd':   train_bd_hist,
            'train_full': train_full_hist,
            'test_full':  test_full_hist,
            'test_step':  test_step_hist,
        }
        if phase['record']:
            data['train_res'] = train_res_hist
        df = pd.DataFrame(data, index=idx)
        df.index.name = 'epoch_in_phase'
        csv_path = os.path.join(path_loss_dir, f"loss_{phase['name']}.csv")
        df.to_csv(csv_path)
        print(f"→ Saved loss history for phase '{phase['name']}' in {csv_path}")

    # Save final model
    torch.save(model, path_model)

