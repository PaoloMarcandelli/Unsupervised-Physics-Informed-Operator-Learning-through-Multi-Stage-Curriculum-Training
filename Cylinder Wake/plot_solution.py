import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from utilities3 import *
from models import Net2d, fluid_model
from spline_models import interpolate_states
from get_params import get_params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from setups import Dataset

import math

def toCuda(x):
	if type(x) is tuple or type(x) is list:
		return [toCuda(xi) for xi in x]
	return x.cuda() 

# Configurazioni base
params       = get_params()
ntrain       = params.ntrain
modes        = params.modes
width        = params.width
epochs       = params.epochs
lbfgs_epochs = params.lbfgs_epochs
T            = params.T


params = get_params()


# Parametri\ 
ntrain      = params.ntrain
ntest       = params.ntest
modes       = params.modes
width       = params.width
bsize       = params.bsize

batch_size  = params.bsize
epochs      = params.epochs
raj_epochs      = params.raj_epochs
lbfgs_epochs= params.lbfgs_epochs
learning_rate   = params.learning_rate
scheduler_step  = params.scheduler_step
scheduler_gamma = params.scheduler_gamma

sub         = 1
S           = 256//sub
T_in        = params.T_in
T           = params.T
step        = params.step
nu          = params.nu 
rf          = params.rf
orders_v    = params.orders_v
n_samples   = params.n_samples

#offset = 30 #Kolmogorov
#offset = 20 # NS nu=1e-4
offset = 40 #NS nu=1e-3

# TRAIN_PATH = 'ns_data_1e-4.mat'
# TEST_PATH  = 'ns_data_1e-4.mat'

TRAIN_PATH = 'ns_data_1e-3.mat'
TEST_PATH  = 'ns_data_1e-3.mat'

# TRAIN_PATH = 'ns_data_Kolmogorov.mat'
# TEST_PATH  = 'ns_data_Kolmogorov.mat'


################################################################################
######################## NS-Kolm FNO ###########################################
################################################################################

path = f'New_ns_fourier{ntrain}_Adam_ep{epochs}_visc{nu}_m{modes}_w{width}_T{T}'

#path = f'kolmogorov_ns_fourier{ntrain}_Adam_ep{epochs}_visc{nu}_m{modes}_w{width}_T{T}'

path_model    = os.path.join('model', path )

path_image     = os.path.join('image', path)
path_loss     = os.path.join('loss', path)

################################################################################
######################## Sup Baseline FNO ######################################
################################################################################

path_raj = 'ns_fourier_2d_rnn'+str(ntrain)+'_ep' + str(raj_epochs) + '_visc' + str(nu) + '_m' + str(modes) + '_w' + str(width)

#path_raj = 'kolmogorov_ns_fourier_2d_rnn'+str(ntrain)+'_ep' + str(120) + '_visc' + str(nu) + '_m' + str(modes) + '_w' + str(width)

path_loss_raj     = os.path.join('loss', path_raj)
os.makedirs(path_loss, exist_ok=True)

################################################################################
######################## Unsup Baseline UNet ###################################
################################################################################

path_unet = f'UNet_ns_fourier{ntrain}_Adam_ep{epochs}_visc{nu}_m{modes}_w{width}_T{T}'
#path_unet = f'UNet_kolmogorov_{ntrain}_Adam_ep{epochs}_visc{nu}_m{modes}_w{width}_T{T}'
path_model_unet    = os.path.join('model', path_unet )
path_loss_unet     = os.path.join('loss', path_unet)
os.makedirs(path_loss_unet, exist_ok=True)

################################################################################
########################  Baseline PINO ########################################
################################################################################

path_pino = 'pino_ns_fourier_2d_rnn'+str(ntrain)+'_ep' + str(raj_epochs) + '_visc' + str(nu) + '_m' + str(modes) + '_w' + str(width)

path_loss_pino     = os.path.join('loss', path_pino)
os.makedirs(path_loss_pino, exist_ok=True)

################################################################################
########################  Baseline Multi-stage PINO ############################
################################################################################

path_pino_ms = 'pino_multi_stage'+str(ntrain)+'_ep' + str(epochs) + '_visc' + str(nu) + '_m' + str(modes) + '_w' + str(width)

path_loss_pino_ms     = os.path.join('loss', path_pino_ms)
os.makedirs(path_loss_pino_ms, exist_ok=True)



################################################################################
######################## Model loading #########################################
################################################################################

model = Net2d(modes, width).cuda()
model = torch.load(path_model, weights_only=False).to(device)

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain, ::sub, ::sub, offset: offset+T_in]
train_u = reader.read_field('u')[:ntrain, ::sub, ::sub,  offset+T_in: offset+T+T_in]

reader = MatReader(TEST_PATH)
test_a  = reader.read_field('u')[-ntest:, ::sub, ::sub,  offset: offset+T_in]
test_u  = reader.read_field('u')[-ntest:, ::sub, ::sub,  offset+T_in: offset+T+T_in]

# Reshape e concat grid
train_a = train_a.reshape(ntrain,S,S,T_in)
test_a  = test_a.reshape(ntest,S,S,T_in)
gridx   = torch.linspace(0,1,S).view(1,S,1,1).repeat(1,1,S,1)
gridy   = torch.linspace(0,1,S).view(1,1,S,1).repeat(1,S,1,1)
gridx_test   = torch.linspace(0,1,S).view(1,S,1,1).repeat(1,1,S,1)
gridy_test   = torch.linspace(0,1,S).view(1,1,S,1).repeat(1,S,1,1)
train_a = torch.cat([gridx.repeat(ntrain,1,1,1), gridy.repeat(ntrain,1,1,1), train_a], dim=-1)
test_a  = torch.cat([gridx_test.repeat(ntest,1,1,1),  gridy_test.repeat(ntest,1,1,1),  test_a],  dim=-1)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

device = torch.device('cuda')
gridx = gridx.to(device)
gridy = gridy.to(device)
gridx_test = gridx_test.to(device)
gridy_test = gridy_test.to(device)

myloss    = LpLoss(size_average=True)
v_size    = np.prod([i+1 for i in orders_v])
pad=1

dataset   = Dataset(batch_size, v_size, S, rf, n_samples, pad)

################################################################################
######################## FNO Resolution ########################################
################################################################################

test_l2_full = 0.0

model.eval()
with torch.no_grad():
    for xx, yy in test_loader:
        xx, yy = toCuda([xx, yy])
        
        # estraggo maschere e stato
        v_mask = toCuda(dataset.ask())  # (B,S,S)
        int_mask = 1 - v_mask
        for t in range(0, T, step):
            im = model(xx)   
            omega, v, grad_omega, lap_omega = interpolate_states(
            im,
            offset=torch.tensor([0.0, 0.0], dtype=torch.float, device=device), orders_v=orders_v
        )                # (1, S, S, 1)
            # preparo input per il passo successivo
            if t == 0:
                pred = omega
            else:
                pred = torch.cat([pred, omega], dim=-1)
        
            xx = torch.cat([
                gridx.repeat(batch_size, 1, 1, 1),
                gridy.repeat(batch_size, 1, 1, 1),
                xx[..., 2+step:], omega

            ], dim=-1)

        
        test_l2_full     += myloss(
    pred.reshape(batch_size, -1),
    yy.reshape(batch_size, -1)
).item()
        
test_full_avg =  test_l2_full/len(test_loader)
print('Test for Resolution FNO:', test_full_avg)

################################################################################
######################## UNet Resolution ########################################
################################################################################

model_unet = fluid_model(params.orders_v).to(device)
model_unet = torch.load(path_model_unet, weights_only=False).to(device)
model_unet.eval()
test_l2_full_unet = 0.0
with torch.no_grad():
    for xx, yy in test_loader:
        xx, yy = toCuda([xx, yy])
        
        # estraggo maschere e stato
        v_mask = toCuda(dataset.ask())  # (B,S,S)
        int_mask = 1 - v_mask
        for t in range(0, T, step):
            im = model_unet(xx)   
            omega, v, grad_omega, lap_omega = interpolate_states(
            im,
            offset=torch.tensor([0.0, 0.0], dtype=torch.float, device=device), orders_v=orders_v
        )                # (1, S, S, 1)
            # preparo input per il passo successivo
            if t == 0:
                pred = omega
            else:
                pred = torch.cat([pred, omega], dim=-1)
        
            xx = torch.cat([
                gridx.repeat(batch_size, 1, 1, 1),
                gridy.repeat(batch_size, 1, 1, 1),
                xx[..., 2+step:], omega

            ], dim=-1)

        
        test_l2_full_unet     += myloss(
    pred.reshape(batch_size, -1),
    yy.reshape(batch_size, -1)
).item()
        
test_full_avg_unet =  test_l2_full_unet/len(test_loader)
print('Test for Resolution UNet:', test_full_avg_unet)


pred_np = pred[0].detach().cpu().numpy() # (T, H, W)
gt_np=yy[0].detach().cpu().numpy()
# zero-based frame indices in your prediction array
frame_idxs  = [0, 4, 9]
# the actual times you want to show above each plot
time_labels = [offset + T_in, offset + T_in +5, offset+T_in+10]

fig, axes = plt.subplots(len(frame_idxs), 3, figsize=(12, 12))
for i, (t, time) in enumerate(zip(frame_idxs, time_labels)):
    gt_f   = gt_np[..., t]
    pred_f = pred_np[..., t]
    error_map = np.abs(gt_f - pred_f)

    # compute the exact same L₂ error as your training loss
    gt_t = torch.from_numpy(gt_f).float().reshape(1,-1).to(device)
    pr_t = torch.from_numpy(pred_f).float().reshape(1,-1).to(device)
    rel_err_lp = myloss(pr_t, gt_t).item()

    # — Column 1 — Ground truth
    ax = axes[i, 0]
    im0 = ax.imshow(gt_f, vmin=gt_np.min(), vmax=gt_np.max(), cmap='viridis')
    ax.set_title(f"GT t = {time}")
    
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.02)

    # — Column 2 — Prediction
    ax = axes[i, 1]
    im1 = ax.imshow(pred_f, vmin=gt_np.min(), vmax=gt_np.max(), cmap='viridis')
    ax.set_title(f"Pred t = {time}")
    
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.02)

    # — Column 3 — Absolute error map + annotation
    ax = axes[i, 2]
    im2 = ax.imshow(error_map, vmin=0, vmax=error_map.max(), cmap='viridis')
    ax.set_title(f"Error t = {time}")
    
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
    ax.set_xlabel(f"Rel L₂ (myloss) = {rel_err_lp:.3e}",
                  labelpad=6,  # adjust this up/down for fine control
                  fontsize=10)
total_error = test_l2_full/len(test_loader)   # o / batch_size se ntest=batch_size
fig.suptitle(f"Total L₂ loss over all timesteps: {total_error:.4e}", fontsize=14)
plt.tight_layout()

# salva nella stessa cartella delle GIF
static_path = os.path.join(path_image, 'comparison_static.png')
os.makedirs(os.path.dirname(static_path), exist_ok=True)
plt.savefig(static_path, dpi=300)
plt.close(fig)
print(f"✔️ Confronto statico salvato in: {static_path}")

path_loss = os.path.join('loss', path)

phases = ['boundary_only_1', 'boundary_only_2', 'full_loss_1', 'full_loss_2', 'full_loss_3']
fields = ['train_bd', 'train_res', 'train_full', 'test_full']

# define a fixed color for each curve
color_map = {
    'train_bd':   'tab:blue',
    'train_res':  'tab:red',
    'train_full': 'tab:orange',
    'test_full':  'tab:green'
}

for phase in phases:
    csv_path = os.path.join(path_loss, f'loss_{phase}.csv')
    if not os.path.isfile(csv_path):
        print(f"⚠️  CSV non trovato per phase {phase}: {csv_path}")
        continue

    df = pd.read_csv(csv_path, index_col='epoch_in_phase')
    xmin, xmax = df.index.min(), df.index.max()

    plt.figure(figsize=(8,5))
    plt.yscale('log')

    # plot each field with its fixed color
    for col in fields:
        if col in df.columns:
            plt.plot(df.index, df[col],
                     label=col,
                     color=color_map[col],
                     linewidth=1.5)

    # annotate final values with dashed gray lines
    final_vals = df.iloc[-1]
    for col in fields:
        if col in final_vals:
            val = final_vals[col]
            # bump only test_full label so it doesn't overlap
            y = val * 1.2 if col == 'test_full' else val
            plt.hlines(y, xmin, xmax,
                       colors='gray', linestyles='--', linewidth=1)
            # use the same curve color for the text
            plt.text(xmax + 1, y,
                     f"{val:.2e}",
                     color=color_map[col],
                     va='center',
                     ha='left',
                     fontsize='small')

    plt.xlabel('Epoch in phase')
    plt.ylabel('Loss')
    plt.title(f'Loss curves – {phase}')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Aggiunge i valori finali ai tick
    yticks = list(plt.yticks()[0])
    for col in fields:
        if col in final_vals:
            yticks.append(final_vals[col])
    plt.yticks(sorted(set(yticks)))

    # Salva
    out_png = os.path.join(path_loss, f'loss_{phase}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✔️  Saved plot for phase '{phase}' in: {out_png}")


def load_test_full_series(loss_dir, phases_list):
    """
    Carica SOLO 'train_full' dai file per-fase loss_<phase>.csv
    Ritorna (x_cum, y_concat, y_last) dove y_last è l'ultimo valore dell'ultima fase trovata.
    """
    cum_epoch = 0
    xs, ys = [], []
    last_val = None
    found_any = False
    for ph in phases_list:
        f = os.path.join(loss_dir, f'loss_{ph}.csv')
        if not os.path.isfile(f):
            continue
        df = pd.read_csv(f, index_col='epoch_in_phase')
        if 'test_full' not in df.columns:
            continue
        y = df['test_full'].to_numpy()
        x = np.arange(1, len(y)+1) + cum_epoch
        xs.append(x)
        ys.append(y)
        cum_epoch += len(y)
        last_val = y[-1]
        found_any = True

    if found_any:
        return np.concatenate(xs), np.concatenate(ys), last_val

    # debug opzionale
    try:
        print(f"⚠️  In {loss_dir} I see files:", os.listdir(loss_dir))
    except Exception:
        pass
    return None, None, None


def load_fno_test_series(loss_dir):
    """
    Carica SOLO 'test_full' da 'loss.csv' del FNO classico.
    Ritorna (x, y, y_last)
    """
    f = os.path.join(loss_dir, 'loss.csv')
    if not os.path.isfile(f):
        print(f"⚠️  No loss.csv in {loss_dir}")
        return None, None, None
    df = pd.read_csv(f, index_col=0)  # indice = 'epoch_in_phase'
    if 'test_full' not in df.columns:
        print(f"⚠️  Column 'test_full' not found in {f}. Available: {list(df.columns)}")
        return None, None, None
    y = df['test_full'].to_numpy()
    x = df.index.to_numpy()
    return x, y, y[-1]


# phases in ordine
phases_list = ['boundary_only_1', 'boundary_only_2', 'full_loss_1', 'full_loss_2', 'full_loss_3']

# spline: train_full per fase
x_spline, y_spline, last_spline = load_test_full_series(path_loss, phases_list)
x_unet_spline, y_unet_spline, last_unet_spline = load_test_full_series(path_loss_unet, phases_list)
# FNO classico: test_full da loss.csv
x_fno, y_fno, last_fno = load_fno_test_series(path_loss_raj)
x_pino, y_pino, last_pino = load_fno_test_series(path_loss_pino)
x_pino_ms, y_pino_ms, last_pino_ms = load_test_full_series(path_loss_pino_ms, phases_list)


# plot
plt.figure(figsize=(8,5))
plt.yscale('log')
ax = plt.gca()
plotted_any = False

if x_spline is not None and y_spline is not None:
    ax.plot(x_spline, y_spline, label='test_full – spline (per-phase)', color='tab:green', linewidth=2.0)
    plotted_any = True
else:
    print(f"⚠️  No per-phase test_full found in {path_loss}")

if x_unet_spline is not None and y_unet_spline is not None:
    ax.plot(x_unet_spline, y_unet_spline, label='UNet: test_full – spline (per-phase)', color='tab:blue', linewidth=2.0)
    plotted_any = True
else:
    print(f"⚠️  No UNET per-phase test_full found in {path_loss}")

if x_fno is not None and y_fno is not None:
    ax.plot(x_fno, y_fno, label='test_full – FNO', color='k', linewidth=2.0, alpha=0.9)
    plotted_any = True
else:
    print(f"⚠️  No FNO test_full in {path_loss_raj}")

if x_pino is not None and y_pino is not None:
    ax.plot(x_pino, y_pino, label='test_full – PINO',color="tab:orange",  linewidth=2.0, alpha=0.9)
    plotted_any = True
else:
    print(f"⚠️  No PINO test_full in {path_loss_pino}")

if x_pino_ms is not None and y_pino_ms is not None:
    ax.plot(x_pino_ms, y_pino_ms, label='test_full – Multi Stage PINO',color="tab:red",  linewidth=2.0, alpha=0.9)
    plotted_any = True
else:
    print(f"⚠️  No Multi-Stage PINO test_full in {path_loss_pino_ms}")

if plotted_any:
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$L_2$')
    ax.set_title('PhIS-FNO vs FNO')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(loc='best')

    # annotazioni finali (come negli altri plot: linea grigia + etichetta colorata)
    xmin, xmax = ax.get_xlim()
    margin = 0.06*(xmax - xmin)
    ax.set_xlim(xmin, xmax + margin)

    if last_spline is not None:
        ax.hlines(last_spline, xmin, xmax, colors='gray', linestyles='--', linewidth=1)
        ax.text(xmax + 0.01*(xmax-xmin), last_spline,
                f"{last_spline:.2e}", color='tab:green',
                va='center', ha='left', fontsize='small')
        print(f"✔️  Final SPLINE test_full = {last_spline:.3e}")

    if last_unet_spline is not None:
        ax.hlines(last_unet_spline, xmin, xmax, colors='gray', linestyles='--', linewidth=1)
        ax.text(xmax + 0.01*(xmax-xmin), last_unet_spline,
                f"{last_unet_spline:.2e}", color='tab:blue',
                va='center', ha='left', fontsize='small')
        print(f"✔️  Final UNET SPLINE test_full = {last_unet_spline:.3e}")

    if last_fno is not None:
        ax.hlines(last_fno, xmin, xmax, colors='gray', linestyles='--', linewidth=1)
        ax.text(xmax + 0.01*(xmax-xmin), last_fno,
                f"{last_fno:.2e}", color='k',
                va='center', ha='left', fontsize='small')
        print(f"✔️  Final FNO test_full    = {last_fno:.3e}")

    if last_pino is not None:
        ax.hlines(last_pino, xmin, xmax, colors="tab:orange", linestyles='--', linewidth=1)
        ax.text(xmax + 0.01*(xmax-xmin), last_pino,
                f"{last_pino:.2e}", color="tab:orange",
                va='center', ha='left', fontsize='small')
        print(f"✔️  Final PINO test_full    = {last_pino:.3e}")

    if last_pino_ms is not None:
        ax.hlines(last_pino_ms, xmin, xmax, colors="tab:red", linestyles='--', linewidth=1)
        ax.text(xmax + 0.01*(xmax-xmin), last_pino_ms,
                f"{last_pino_ms:.2e}", color="tab:red",
                va='center', ha='left', fontsize='small')
        print(f"✔️  Final MS-PINO test_full    = {last_pino_ms:.3e}")
    

    # salvataggio
    os.makedirs(path_loss, exist_ok=True)
    cmp_png = os.path.join(path_loss, 'trainfull_vs_fno_testfull.png')
    plt.tight_layout()
    plt.savefig(cmp_png, dpi=300)
    plt.close()
    print(f"✔️  Saved comparison plot in: {cmp_png}")
else:
    plt.close()


# ============================
# GIF: GT vs Pred vs Error (test batch[0], tutti i timesteps)
# ============================
import os, numpy as np, torch
import matplotlib
matplotlib.use("Agg")           # sicuro in headless
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from utilities3 import LpLoss
from spline_models import interpolate_states

def _rollout_first_sample(model, xx, yy, gridx, gridy, orders_v, device, step=1):
    """
    Esegue il rollout autoregressivo come nel tuo codice e restituisce:
      pred_0 : (S, S, T) prediction vorticità per il primo sample della batch
      gt_0   : (S, S, T) ground truth corrispondente
    """
    model.eval()
    B, S, _, T_in = xx.shape[-4], xx.shape[-3], xx.shape[-2], xx.shape[-1]  # xx: (B,S,S,2+T_in)
    T = yy.shape[-1]  # orizzonte di test da predire

    # lavoriamo su una copia, per non distruggere xx
    xx_work = xx.clone()

    pred = None
    with torch.no_grad():
        for t in range(0, T, step):
            im = model(xx_work)  # (B, S, S, v_size)
            omega, v, grad_omega, lap_omega = interpolate_states(
                im,
                offset=torch.tensor([0.0, 0.0], dtype=torch.float, device=device),
                orders_v=orders_v
            )  # omega: (B, S, S, 1)

            pred = omega if pred is None else torch.cat([pred, omega], dim=-1)  # (B,S,S,t+1)

            # prepara input per il passo successivo (griglia + finestre temporali + nuova omega)
            xx_work = torch.cat([
                gridx.repeat(B, 1, 1, 1),
                gridy.repeat(B, 1, 1, 1),
                xx_work[..., 2+step:],  # shift della finestra temporale
                omega
            ], dim=-1)

    # estraiamo il primo elemento della batch e portiamo su CPU
    pred_0 = pred[0].detach().cpu().numpy()  # (S,S,T)
    gt_0   = yy[0].detach().cpu().numpy()    # (S,S,T)
    return pred_0, gt_0

def save_test_comparison_gif(
    model, test_loader, gridx, gridy, orders_v, device,
    out_path="image/test_pred_vs_gt.gif", fps=12, dpi=120,
    cmap="viridis", step=1, base_time=None
):
    """
    Crea una GIF 1x3 (GT | Pred | Error) sul primo elemento del primo batch del test_loader.
    - base_time: se vuoi etichettare i frame con l’indice assoluto (es. 40+T_in),
                 passa un intero; altrimenti mostra t=0,1,2,...
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # prendi il primo batch del test_loader
    xx, yy = next(iter(test_loader))
    xx, yy = xx.to(device), yy.to(device)

    # rollout
    pred_np, gt_np = _rollout_first_sample(model, xx, yy, gridx, gridy, orders_v, device, step=step)
    # reshape per iterare comodamente sul tempo: (T, S, S)
    pred_T = np.moveaxis(pred_np, -1, 0)
    gt_T   = np.moveaxis(gt_np,   -1, 0)
    err_T  = np.abs(gt_T - pred_T)

    T = gt_T.shape[0]
    vmin, vmax     = gt_T.min(), gt_T.max()       # stessa scala di colore di GT per GT/Pred
    err_vmin, err_vmax = 0.0, err_T.max() + 1e-12 # scala coerente per l'errore

    myloss = LpLoss(size_average=True)

    # figura 1x3
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ax_gt, ax_pr, ax_er = axes

    im_gt = ax_gt.imshow(gt_T[0], vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    ax_gt.set_title("GT")
    cb0 = fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    im_pr = ax_pr.imshow(pred_T[0], vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    ax_pr.set_title("Pred")
    cb1 = fig.colorbar(im_pr, ax=ax_pr, fraction=0.046, pad=0.04)

    im_er = ax_er.imshow(err_T[0], vmin=err_vmin, vmax=err_vmax, cmap=cmap, interpolation="nearest")
    ax_er.set_title("Error (|GT-Pred|)")
    cb2 = fig.colorbar(im_er, ax=ax_er, fraction=0.046, pad=0.04)

    # testo per Rel L2
    txt = ax_er.text(
        0.02, -0.08, "", transform=ax_er.transAxes, fontsize=10, va="top"
    )

    def _rel_l2_frame(k):
        gt = torch.from_numpy(gt_T[k]).float().reshape(1, -1)
        pr = torch.from_numpy(pred_T[k]).float().reshape(1, -1)
        return myloss(pr, gt).item()

    def _time_label(k):
        return (base_time + k*step) if base_time is not None else k

    def update(k):
        im_gt.set_data(gt_T[k])
        im_pr.set_data(pred_T[k])
        im_er.set_data(err_T[k])

        rel_l2 = _rel_l2_frame(k)
        ax_gt.set_title(f"GT — t = {_time_label(k)}")
        ax_pr.set_title(f"Pred — t = {_time_label(k)}")
        ax_er.set_title(f"Error — t = {_time_label(k)}")
        txt.set_text(f"Rel L₂ (LpLoss) = {rel_l2:.3e}")
        return (im_gt, im_pr, im_er, txt)

    ani = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"✔️ GIF salvata in: {out_path} (frames={T})")

# ======= ESEMPIO D’USO =======
# base_time=40+T_in perché i tuoi test sono su [40+T_in : 40+T+T_in]
save_test_comparison_gif(
    model=model,
    test_loader=test_loader,
    gridx=gridx, gridy=gridy,
    orders_v=orders_v,
    device=device,
    out_path=f"image/test_pred_vs_gt_nu{nu}.gif",
    fps=12, dpi=120, cmap="viridis",
    step=step,
    base_time=offset + T_in
)
