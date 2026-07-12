"""
plot_solution.py — Burger 1D: confronto loss e plot soluzioni.

Genera le figure in figures/:
  1. loss_phisfno.png       — tutte le strategie per PhIS-FNO
  2. loss_pino_fd.png       — tutte le strategie per PINO-FD
  3. loss_pino_spectral.png — tutte le strategie per PINO-Spectral (FFT)
  4. loss_combined.png      — cross-model MS comparison (tutti e 3)
  5. loss_sidebyside.png    — pannello 1×3 PhIS-FNO | PINO-FD | PINO-Spectral
  6. solution_phisfno.png   — IC / GT / Pred / Error del modello phisfno MS
"""

import os, math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utilities3 import MatReader, LpLoss
from spline_models1d import interpolate_states
from models import Net1d

# ════════════════════════════════════════════════════════════════════════════
# CONFIG  — deve corrispondere ai parametri usati in train_model.py / run_all.sh
# ════════════════════════════════════════════════════════════════════════════

ntrain      = 90
ntest       = 10
sub         = 4
nu          = 0.1
modes       = 32
width       = 64
epochs      = 100    # per ms / ms_nr
ss_epochs   = 200    # per singlestage
sup_epochs  = 300    # per supervised


OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# PATH BUILDER  — usa la convenzione di train_model.py
# ════════════════════════════════════════════════════════════════════════════

def _ep(training):
    return {
        'multistage':    epochs,
        'multistage_nr': epochs,
        'singlestage':   ss_epochs,
        'supervised':    sup_epochs,
    }[training]

def loss_dir(model, training, seed=None):
    """seed=None → nessun suffisso (run da train_model.py);
       seed=<int> → suffisso _s{seed} (run da train_model_ablation.py)."""
    tag = {'multistage': 'ms', 'multistage_nr': 'ms_nr',
           'singlestage': 'ss', 'supervised': 'sup'}[training]
    suffix = f'_s{seed}' if seed is not None else ''
    folder = f'Burger1d_{model}_{tag}_{ntrain}_Adam_ep{_ep(training)}_visc{nu}_m{modes}_w{width}{suffix}'
    return os.path.join('loss', folder)

def model_path(model, training, seed=None):
    """seed=None → nessun suffisso (run da train_model.py);
       seed=<int> → suffisso _s{seed} (run da train_model_ablation.py)."""
    tag = {'multistage': 'ms', 'multistage_nr': 'ms_nr',
           'singlestage': 'ss', 'supervised': 'sup'}[training]
    suffix = f'_s{seed}' if seed is not None else ''
    folder = f'Burger1d_{model}_{tag}_{ntrain}_Adam_ep{_ep(training)}_visc{nu}_m{modes}_w{width}{suffix}'
    return os.path.join('model', folder)

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════

PHASE_NAMES = ['full_loss_1', 'full_loss_2', 'full_loss_3', 'supervised']

def load_series(loss_dir_path):
    """
    Carica test_full concatenando tutti i file loss_<phase>.csv trovati in ordine.
    Ritorna (x_cum, y, plateau_mean, plateau_std) oppure (None,...) se non trovato.
    """
    xs, ys = [], []
    cum = 0
    for ph in PHASE_NAMES:
        f = os.path.join(loss_dir_path, f'loss_{ph}.csv')
        if not os.path.isfile(f):
            continue
        df = pd.read_csv(f, index_col='epoch_in_phase')
        if 'test_full' not in df.columns:
            continue
        y = df['test_full'].to_numpy()
        xs.append(np.arange(1, len(y) + 1) + cum)
        ys.append(y)
        cum += len(y)

    if not xs:
        return None, None, None, None

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    pm, ps = _plateau_stats(y_all)
    return x_all, y_all, pm, ps


def _plateau_stats(y, tail=50):
    """Mean ± std degli ultimi `tail` punti."""
    seg = y[-min(tail, len(y)):]
    return float(np.mean(seg)), float(np.std(seg))


def phase_lengths(training):
    """Lunghezza in epoche di ogni fase (per i divisori verticali)."""
    if training in ('multistage', 'multistage_nr'):
        return [epochs, epochs, epochs]
    elif training == 'singlestage':
        return [ss_epochs]
    else:
        return [sup_epochs]

# ════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def rolling_stats_log(y, win=11, eps=1e-12):
    y = np.maximum(np.asarray(y, dtype=float), eps)
    if len(y) < win:
        win = max(3, len(y) | 1)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    yp  = np.pad(np.log10(y), (pad, pad), mode='edge')
    k   = np.ones(win) / win
    m   = np.convolve(yp, k, mode='valid')
    s   = np.sqrt(np.maximum(0, np.convolve(yp**2, k, mode='valid') - m**2))
    return 10**m, 10**(m - s), 10**(m + s)


def plot_curve(ax, x, y, color, label, lw=2.0, win=11, zorder=3):
    if x is None:
        return
    ymean, ylow, yup = rolling_stats_log(y, win=win)
    ax.plot(x, ymean, color=color, linewidth=lw, label=label, zorder=zorder)
    ax.fill_between(x, np.maximum(ylow, 1e-12), yup,
                    color=color, alpha=0.18, linewidth=0, zorder=zorder - 1)


def plot_any(ax, entry, color, label, lw=2.0, win=11, zorder=3):
    """Wrapper: estrae (x, y) dalla 5-tupla e chiama plot_curve."""
    x, y = entry[0], entry[1]
    plot_curve(ax, x, y, color, label, lw=lw, win=win, zorder=zorder)


def add_phase_dividers(ax, training):
    lengths = phase_lengths(training)
    pos = np.cumsum(lengths)[:-1]
    for p in pos:
        ax.axvline(p, color='k', linewidth=0.7, linestyle='--', alpha=0.6)


def finalize_ax(ax, title):
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(r'Relative $L_2$ test error', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)


def print_plateau(label, pm, ps):
    if pm is not None:
        print(f'  {label:35s}  plateau = {pm:.3e} ± {ps:.3e}')
    else:
        print(f'  {label:35s}  not found')

# ════════════════════════════════════════════════════════════════════════════
# COLOUR SCHEME — colore = strategia di training, uguale per tutti i modelli
# ════════════════════════════════════════════════════════════════════════════
C_MS    = '#00C853'   # verde  — multistage (reset)
C_MSNR  = '#0062B2'   # blu    — multistage no-reset
C_SS    = '#7C4DFF'   # lilla  — singlestage
C_SUP   = '#000000'   # nero   — supervised

# ════════════════════════════════════════════════════════════════════════════
# LOAD ALL SERIES
# ════════════════════════════════════════════════════════════════════════════

PHISFNO_SEEDS = [0, 11, 22, 33]

def plateau_multiseed(model, training, seeds, tail=50):
    """
    Per ogni seed carica la serie e ne calcola il valore di plateau.
    Ritorna (plateaus_per_seed, plateau_mean, plateau_std).
    plateaus_per_seed: dict {seed: plateau_value}  (None se mancante)
    """
    per_seed = {}
    for s in seeds:
        ld = loss_dir(model, training, seed=s)
        x, y, pm, ps = load_series(ld)
        status = 'OK' if x is not None else 'MISSING'
        print(f'[{status:7s}] {model}_{training} seed={s:2d}  {ld}')
        per_seed[s] = pm   # pm = mean degli ultimi `tail` punti (o None)
    vals = [v for v in per_seed.values() if v is not None]
    if not vals:
        return per_seed, None, None
    arr = np.array(vals)
    return per_seed, float(arr.mean()), float(arr.std())


data = {}
multiseed_stats = {}   # {key: (per_seed, plateau_mean, plateau_std)}

# PhIS-FNO: plot da train_model.py (senza seed nel nome); statistiche su tutti i seed
print('\n--- Caricamento PhIS-FNO ---')
for training in ('multistage', 'multistage_nr', 'singlestage', 'supervised'):
    key = f'phisfno_{training}'
    # curva senza seed (da train_model.py)
    ld = loss_dir('phisfno', training, seed=None)
    x, y, pm, ps = load_series(ld)
    data[key] = (x, y, None, pm, ps)
    # plateau su tutti i seed per le statistiche
    per_seed, p_mean, p_std = plateau_multiseed('phisfno', training, PHISFNO_SEEDS)
    multiseed_stats[key] = (per_seed, p_mean, p_std)

# PINO-FD e PINO-Spectral: da train_model.py (senza seed nel nome)
print('\n--- Caricamento PINO-FD / PINO-Spectral ---')
for model in ('pino_fd', 'pino_spectral'):
    for training in ('multistage', 'multistage_nr', 'singlestage', 'supervised'):
        key = f'{model}_{training}'
        ld  = loss_dir(model, training, seed=None)
        x, y, pm, ps = load_series(ld)
        data[key] = (x, y, None, pm, ps)
        status = 'OK' if x is not None else 'MISSING'
        print(f'[{status:7s}] {key:40s}  {ld}')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — PhIS-FNO: tutte le strategie  (banda = std tra seed)
# ════════════════════════════════════════════════════════════════════════════

print('\n--- PhIS-FNO ---')
fig1, ax1 = plt.subplots(figsize=(8, 5))

e = data['phisfno_multistage']
plot_any(ax1, e, C_MS,   'MS (reset)',    win=15)
print_plateau('phisfno — multistage',    e[3], e[4])
if e[0] is not None: add_phase_dividers(ax1, 'multistage')

e = data['phisfno_multistage_nr']
plot_any(ax1, e, C_MSNR, 'MS no-reset',  win=15)
print_plateau('phisfno — multistage_nr', e[3], e[4])

e = data['phisfno_singlestage']
plot_any(ax1, e, C_SS,   'Single-stage', win=15)
print_plateau('phisfno — singlestage',   e[3], e[4])

e = data['phisfno_supervised']
plot_any(ax1, e, C_SUP,  'Supervised',   win=15, lw=1.8)
print_plateau('phisfno — supervised',    e[3], e[4])

finalize_ax(ax1, 'PhIS-FNO — Training strategy comparison')
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'loss_phisfno.png'), dpi=200)
plt.close(fig1)
print(f'\n→ Saved: {OUT_DIR}/loss_phisfno.png')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — PINO-FD: tutte le strategie
# ════════════════════════════════════════════════════════════════════════════

print('\n--- PINO-FD ---')
fig2, ax2 = plt.subplots(figsize=(8, 5))

e = data['pino_fd_multistage']
plot_any(ax2, e, C_MS,   'MS (reset)',    win=15)
print_plateau('pino_fd — multistage',    e[3], e[4])
if e[0] is not None: add_phase_dividers(ax2, 'multistage')

e = data['pino_fd_multistage_nr']
plot_any(ax2, e, C_MSNR, 'MS no-reset',  win=15)
print_plateau('pino_fd — multistage_nr', e[3], e[4])

e = data['pino_fd_singlestage']
plot_any(ax2, e, C_SS,   'Single-stage', win=15)
print_plateau('pino_fd — singlestage',   e[3], e[4])

e = data['pino_fd_supervised']
plot_any(ax2, e, C_SUP,  'Supervised',   win=15, lw=1.8)
print_plateau('pino_fd — supervised',    e[3], e[4])

finalize_ax(ax2, 'PINO-FD — Training strategy comparison')
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'loss_pino_fd.png'), dpi=200)
plt.close(fig2)
print(f'→ Saved: {OUT_DIR}/loss_pino_fd.png')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — PINO-Spectral: tutte le strategie
# ════════════════════════════════════════════════════════════════════════════

print('\n--- PINO-Spectral ---')
fig3, ax3 = plt.subplots(figsize=(8, 5))

e = data['pino_spectral_multistage']
plot_any(ax3, e, C_MS,   'MS (reset)',    win=15)
print_plateau('pino_spectral — multistage',    e[3], e[4])
if e[0] is not None: add_phase_dividers(ax3, 'multistage')

e = data['pino_spectral_multistage_nr']
plot_any(ax3, e, C_MSNR, 'MS no-reset',  win=15)
print_plateau('pino_spectral — multistage_nr', e[3], e[4])

e = data['pino_spectral_singlestage']
plot_any(ax3, e, C_SS,   'Single-stage', win=15)
print_plateau('pino_spectral — singlestage',   e[3], e[4])

e = data['pino_spectral_supervised']
plot_any(ax3, e, C_SUP,  'Supervised',   win=15, lw=1.8)
print_plateau('pino_spectral — supervised',    e[3], e[4])

finalize_ax(ax3, 'PINO-Spectral — Training strategy comparison')
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'loss_pino_spectral.png'), dpi=200)
plt.close(fig3)
print(f'→ Saved: {OUT_DIR}/loss_pino_spectral.png')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Cross-model MS comparison (tutti e 3, solo multistage + supervised)
# ════════════════════════════════════════════════════════════════════════════

print('\n--- Cross-model ---')
fig4, ax4 = plt.subplots(figsize=(9, 5))

e = data['phisfno_multistage']
plot_any(ax4, e, C_MS,   'PhIS-FNO MS',          win=15)
if e[0] is not None: add_phase_dividers(ax4, 'multistage')

e = data['phisfno_multistage_nr']
plot_any(ax4, e, C_MSNR, 'PhIS-FNO MS no-reset', win=15)

e = data['pino_fd_multistage']
plot_any(ax4, e, C_MS,   'PINO-FD MS',           win=15)

e = data['pino_fd_multistage_nr']
plot_any(ax4, e, C_MSNR, 'PINO-FD MS no-reset',  win=15)

e = data['pino_spectral_multistage']
plot_any(ax4, e, C_MS,   'PINO-Spectral MS',     win=15)

e = data['pino_spectral_multistage_nr']
plot_any(ax4, e, C_MSNR, 'PINO-Spectral MS no-reset', win=15)

e = data['phisfno_supervised']
plot_any(ax4, e, C_SUP,  'Supervised (PhIS-FNO)', win=15, lw=1.5)

finalize_ax(ax4, 'PhIS-FNO vs PINO-FD vs PINO-Spectral — MS comparison')
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'loss_combined.png'), dpi=200)
plt.close(fig4)
print(f'→ Saved: {OUT_DIR}/loss_combined.png')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — 1×3 side-by-side: PhIS-FNO | PINO-FD | PINO-Spectral
# ════════════════════════════════════════════════════════════════════════════

fig5, (axL, axM, axR) = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

# --- Left: PhIS-FNO ---
for (training, color, label) in [
    ('multistage',    C_MS,   'MS (reset)'),
    ('multistage_nr', C_MSNR, 'MS no-reset'),
    ('singlestage',   C_SS,   'Single-stage'),
    ('supervised',    C_SUP,  'Supervised'),
]:
    e  = data[f'phisfno_{training}']
    lw = 1.8 if training == 'supervised' else 2.0
    plot_any(axL, e, color, label, win=15, lw=lw)
if data['phisfno_multistage'][0] is not None:
    add_phase_dividers(axL, 'multistage')
finalize_ax(axL, 'PhIS-FNO')

# --- Middle: PINO-FD ---
for (training, color, label) in [
    ('multistage',    C_MS,   'MS (reset)'),
    ('multistage_nr', C_MSNR, 'MS no-reset'),
    ('singlestage',   C_SS,   'Single-stage'),
    ('supervised',    C_SUP,  'Supervised'),
]:
    e  = data[f'pino_fd_{training}']
    lw = 1.8 if training == 'supervised' else 2.0
    plot_any(axM, e, color, label, win=15, lw=lw)
if data['pino_fd_multistage'][0] is not None:
    add_phase_dividers(axM, 'multistage')
axM.set_ylabel('')
finalize_ax(axM, 'PINO-FD')

# --- Right: PINO-Spectral ---
for (training, color, label) in [
    ('multistage',    C_MS,   'MS (reset)'),
    ('multistage_nr', C_MSNR, 'MS no-reset'),
    ('singlestage',   C_SS,   'Single-stage'),
    ('supervised',    C_SUP,  'Supervised'),
]:
    e  = data[f'pino_spectral_{training}']
    lw = 1.8 if training == 'supervised' else 2.0
    plot_any(axR, e, color, label, win=15, lw=lw)
if data['pino_spectral_multistage'][0] is not None:
    add_phase_dividers(axR, 'multistage')
axR.set_ylabel('')
finalize_ax(axR, 'PINO-Spectral (FFT)')

fig5.suptitle('Burger 1D — Training strategy comparison', fontsize=14, fontweight='bold', y=1.02)
fig5.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, 'loss_sidebyside.png'), dpi=200, bbox_inches='tight')
plt.close(fig5)
print(f'→ Saved: {OUT_DIR}/loss_sidebyside.png')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Solution plot (PhIS-FNO MS, campione di test)
# ════════════════════════════════════════════════════════════════════════════

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S      = (2**13) // sub
myloss = LpLoss(size_average=True)

reader = MatReader('dataset/burgers1d_dataset.mat')
x_data = reader.read_field('a')[:, ::sub]
y_data = reader.read_field('u')[:, -1, ::sub]
x_test = x_data[-ntest:]
y_test = y_data[-ntest:]
grid   = torch.tensor(
    np.linspace(0, 2*math.pi, S).reshape(1, S, 1), dtype=torch.float)
x_test_in = torch.cat([x_test.reshape(ntest, S, 1), grid.repeat(ntest, 1, 1)], dim=2)

# Carica il modello phisfno ms (se disponibile)
mp = model_path('phisfno', 'multistage')
if os.path.isfile(mp):
    model = torch.load(mp, map_location=device, weights_only=False)
    model.eval()

    pred = torch.zeros(y_test.shape)
    loader1 = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_in, y_test),
        batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (x, y) in enumerate(loader1):
            x, y = x.to(device), y.to(device)
            im   = model(x)
            u, _, _ = interpolate_states(
                im, offset=torch.tensor([0.0], device=device), orders_v=[2])
            pred[i] = u.cpu()

    # Campione indice 1
    idx  = 1
    ic0  = x_test_in[idx, :, 0].numpy()
    gt0  = y_test[idx].numpy()
    pr0  = pred[idx].detach().numpy()
    err0 = pr0 - gt0

    l2n  = lambda a: float(np.sqrt(np.mean(a**2)))
    xax  = np.linspace(0, 2*math.pi, S)

    fig6, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True)
    pairs = [
        (ic0,  f'IC  ‖u₀‖₂={l2n(ic0):.3e}',  'tab:gray'),
        (gt0,  f'GT  ‖u‖₂={l2n(gt0):.3e}',   'tab:blue'),
        (pr0,  f'Pred ‖û‖₂={l2n(pr0):.3e}',  C_MS),
        (err0, f'Error ‖e‖₂={l2n(err0):.3e}', 'tab:red'),
    ]
    for ax, (val, title, col) in zip(axes, pairs):
        ax.plot(xax, val, color=col, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('u')
    fig6.suptitle('PhIS-FNO MS — test sample', fontsize=13, fontweight='bold')
    fig6.tight_layout()
    fig6.savefig(os.path.join(OUT_DIR, 'solution_phisfno.png'), dpi=200)
    plt.close(fig6)
    print(f'→ Saved: {OUT_DIR}/solution_phisfno.png')

    # ── FIGURE 7 — IC + GT + Pred sovrapposti | Error ──────────────────────
    fig7, (ax7L, ax7R) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: IC, GT, Pred sovrapposti
    ax7L.plot(xax, ic0,  color='tab:blue',   linewidth=2.0,
              label=r'IC  $u_0(x)$')
    ax7L.plot(xax, gt0,  color='tab:orange',  linewidth=2.0,
              label=r'GT  $u(x,T)$')
    ax7L.plot(xax, pr0,  color='tab:green',   linewidth=2.0,
              linestyle='--', label=r'Pred  $\hat{u}(x,T)$')
    ax7L.set_xlabel('$x$', fontsize=14)
    ax7L.set_ylabel('$u$', fontsize=14)
    ax7L.tick_params(labelsize=12)
    ax7L.legend(fontsize=13, framealpha=0.9)
    ax7L.grid(True, alpha=0.3)

    # Right: Error
    ax7R.plot(xax, err0, color='tab:red', linewidth=2.0)
    ax7R.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax7R.set_xlabel('$x$', fontsize=14)
    ax7R.set_ylabel(r'$\hat{u} - u$', fontsize=14)
    ax7R.tick_params(labelsize=12)
    ax7R.grid(True, alpha=0.3)
    fig7.tight_layout()
    fig7.savefig(os.path.join(OUT_DIR, 'solution_overlay_phisfno.png'), dpi=200)
    plt.close(fig7)
    print(f'→ Saved: {OUT_DIR}/solution_overlay_phisfno.png')
    print(f'  Pointwise error — mean: {float(np.mean(np.abs(err0))):.3e}  |  L2: {l2n(err0):.3e}  |  max: {float(np.max(np.abs(err0))):.3e}')

else:
    print(f'⚠  Model not found: {mp}  — solution plot skipped.')

# ════════════════════════════════════════════════════════════════════════════
# MULTI-SEED ANALYSIS — PhIS-FNO plateau statistics across seeds
# ════════════════════════════════════════════════════════════════════════════

TRAINING_LABELS = {
    'multistage':    'MS (reset)   ',
    'multistage_nr': 'MS no-reset  ',
    'singlestage':   'Single-stage ',
    'supervised':    'Supervised   ',
}

print('\n' + '═' * 62)
print(' PhIS-FNO — Multi-seed plateau analysis')
print(f' Seeds: {PHISFNO_SEEDS}   (tail=50 epochs)')
print('═' * 62)
print(f'  {"Strategy":<16}  {"mean":>10}  {"std":>10}  {"per seed"}')
print('─' * 62)
for training in ('multistage', 'multistage_nr', 'singlestage', 'supervised'):
    key = f'phisfno_{training}'
    per_seed, p_mean, p_std = multiseed_stats[key]
    label = TRAINING_LABELS[training]
    if p_mean is not None:
        per_str = '  '.join(
            f's{s}={per_seed[s]:.3e}' if per_seed[s] is not None else f's{s}=N/A'
            for s in PHISFNO_SEEDS
        )
        print(f'  {label}  {p_mean:10.3e}  {p_std:10.3e}  {per_str}')
    else:
        print(f'  {label}  {"N/A":>10}  {"N/A":>10}  (no data)')
print('═' * 62)

# ════════════════════════════════════════════════════════════════════════════
# PHASE LOSS PLOTS — PhIS-FNO MS, seed=0
# Una figura per fase con train_bd, train_res, test_full.
# Salvate in image/phases/  +  summary CSV con valori medi finali.
# ════════════════════════════════════════════════════════════════════════════

PHASES_DIR = os.path.join(OUT_DIR, 'phases')
os.makedirs(PHASES_DIR, exist_ok=True)

_ms_loss_dir = loss_dir('phisfno', 'multistage', seed=None)
_phase_files = ['full_loss_1', 'full_loss_2', 'full_loss_3']
_phase_labels = ['Phase 1  (λ_sup=0.8, λ_res=0.5)',
                 'Phase 2  (λ_sup=0.5, λ_res=1.0)',
                 'Phase 3  (λ_sup=0.5, λ_res=1.5)']

_TAIL = 20   # ultimi N epoch per calcolare il valore medio a convergenza

summary_rows = []   # accumulatore per il CSV

for ph_name, ph_label in zip(_phase_files, _phase_labels):
    csv_f = os.path.join(_ms_loss_dir, f'loss_{ph_name}.csv')
    if not os.path.isfile(csv_f):
        print(f'⚠  Phase file not found: {csv_f} — skipped.')
        continue

    df = pd.read_csv(csv_f, index_col='epoch_in_phase')
    epochs_ax = df.index.to_numpy()

    fig_ph, ax_ph = plt.subplots(figsize=(8, 5))

    for col, color, lbl in [
        ('train_bd',  '#E65100', 'Boundary loss (train)'),
        ('train_res', '#1565C0', 'Residual loss (train)'),
        ('test_full', '#2E7D32', 'Test L₂'),
    ]:
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        ymean, ylow, yup = rolling_stats_log(y, win=11)
        ax_ph.plot(epochs_ax, ymean, color=color, linewidth=2.0, label=lbl)
        ax_ph.fill_between(epochs_ax, np.maximum(ylow, 1e-12), yup,
                           color=color, alpha=0.15, linewidth=0)

        # valore medio a convergenza (ultimi _TAIL epoch)
        tail_mean = float(np.mean(y[-min(_TAIL, len(y)):]))
        tail_std  = float(np.std(y[-min(_TAIL, len(y)):]))
        summary_rows.append({
            'phase':      ph_name,
            'loss':       col,
            'tail_mean':  tail_mean,
            'tail_std':   tail_std,
        })

    ax_ph.set_yscale('log')
    ax_ph.set_xlabel('Epoch in phase', fontsize=16)
    ax_ph.set_ylabel('Loss', fontsize=16)
    ax_ph.set_title(f'PhIS-FNO MS — {ph_label}', fontsize=17, fontweight='bold')
    ax_ph.tick_params(axis='both', labelsize=14)
    ax_ph.grid(True, which='both', ls=':', alpha=0.4)
    ax_ph.legend(fontsize=14, framealpha=0.9)
    fig_ph.tight_layout()

    out_png = os.path.join(PHASES_DIR, f'{ph_name}.png')
    fig_ph.savefig(out_png, dpi=200)
    plt.close(fig_ph)
    print(f'→ Saved: {out_png}')

# CSV summary
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    # pivot: righe = phase, colonne = loss
    summary_pivot = summary_df.pivot(index='phase', columns='loss',
                                     values=['tail_mean', 'tail_std'])
    summary_pivot.columns = [f'{v}_{c}' for v, c in summary_pivot.columns]
    summary_pivot = summary_pivot.reindex(_phase_files)
    csv_out = os.path.join(PHASES_DIR, 'phase_convergence.csv')
    summary_pivot.to_csv(csv_out)
    print(f'→ Saved: {csv_out}')

    print('\n--- Phase convergence summary (PhIS-FNO MS, seed=0) ---')
    print(f'  {"Phase":<14}  {"train_bd":>12}  {"train_res":>12}  {"test_full":>12}')
    print('  ' + '─' * 56)
    for ph in _phase_files:
        row = summary_df[summary_df['phase'] == ph]
        vals = {r['loss']: r['tail_mean'] for _, r in row.iterrows()}
        print(f'  {ph:<14}  '
              f'{vals.get("train_bd",  float("nan")):12.3e}  '
              f'{vals.get("train_res", float("nan")):12.3e}  '
              f'{vals.get("test_full", float("nan")):12.3e}')

# ════════════════════════════════════════════════════════════════════════════
# OPTIMIZER DIAGNOSTICS — PhIS-FNO MS
# Plots saved in figures/optimizer/
# ════════════════════════════════════════════════════════════════════════════

import glob as _glob

OPT_DIR = os.path.join(OUT_DIR, 'optimizer')
os.makedirs(OPT_DIR, exist_ok=True)

_opt_loss_dir = loss_dir('phisfno', 'multistage', seed=None)
# check3 / R_lb are theoretical no-reset diagnostics (see paper): they need
# the accumulated m_{t-1}, v_{t-1} that only exist in the no-reset run.
# train_model.py always writes with a _s{seed} suffix (default seed=0).
_swdiag_loss_dir = loss_dir('phisfno', 'multistage_nr', seed=0)

_PRETTY = {
    'full_loss_1': 'Stage 1',
    'full_loss_2': 'Stage 2',
    'full_loss_3': 'Stage 3',
    'boundary_only_1': 'Stage 0',
}

def _stage_order(name):
    return {'boundary_only_1': 0, 'full_loss_1': 1,
            'full_loss_2': 2,     'full_loss_3': 3}.get(name, 999)

# ── A: eta_eff_mean over training ────────────────────────────────────────────
_eta_csv = os.path.join(_opt_loss_dir, 'eta_term_history.csv')
if os.path.isfile(_eta_csv):
    df_eta = pd.read_csv(_eta_csv)
    if 'eta_eff_mean' not in df_eta.columns and {'term_mean', 'lr'} <= set(df_eta.columns):
        df_eta['eta_eff_mean'] = df_eta['lr'] * df_eta['term_mean']
    if 'epoch_global' not in df_eta.columns:
        df_eta['epoch_global'] = range(1, len(df_eta) + 1)

    # phase boundaries: punti in cui epoch_in_phase == 1 (tranne il primo)
    boundaries = []
    if 'epoch_in_phase' in df_eta.columns:
        mask = (df_eta['epoch_in_phase'] == 1) & (df_eta['epoch_global'] > df_eta['epoch_global'].min())
        boundaries = df_eta.loc[mask, 'epoch_global'].tolist()

    # carica anche ms_nr per confronto
    _eta_csv_nr = os.path.join(loss_dir('phisfno', 'multistage_nr', seed=None), 'eta_term_history.csv')
    df_eta_nr = None
    if os.path.isfile(_eta_csv_nr):
        df_eta_nr = pd.read_csv(_eta_csv_nr)
        if 'eta_eff_mean' not in df_eta_nr.columns and {'term_mean', 'lr'} <= set(df_eta_nr.columns):
            df_eta_nr['eta_eff_mean'] = df_eta_nr['lr'] * df_eta_nr['term_mean']
        if 'epoch_global' not in df_eta_nr.columns:
            df_eta_nr['epoch_global'] = range(1, len(df_eta_nr) + 1)

    fig_eta, ax_eta = plt.subplots(figsize=(10, 4))
    ax_eta.plot(df_eta['epoch_global'], df_eta['eta_eff_mean'],
                color='#1f4e79', linewidth=2.0, label='MS (reset)')
    if df_eta_nr is not None:
        ax_eta.plot(df_eta_nr['epoch_global'], df_eta_nr['eta_eff_mean'],
                    color='tab:red', linewidth=2.0, label='MS (no reset)')
    for b in boundaries:
        ax_eta.axvline(b, ls='--', lw=1.0, color='gray', alpha=0.6)
    ax_eta.set_yscale('log')
    ax_eta.set_xlabel('Epoch', fontsize=14)
    ax_eta.set_ylabel(r'$\bar{\eta}_{\mathrm{eff}}$', fontsize=14)
    ax_eta.tick_params(labelsize=12)
    ax_eta.legend(fontsize=13, framealpha=0.9)
    ax_eta.grid(True, alpha=0.3)
    fig_eta.tight_layout()
    _out = os.path.join(OPT_DIR, 'eta_eff_mean.png')
    fig_eta.savefig(_out, dpi=200)
    plt.close(fig_eta)
    print(f'→ Saved: {_out}')
else:
    print(f'⚠  eta_term_history.csv not found in {_opt_loss_dir} — eta plot skipped.')

# ── B: R_lb per layer (scatter semilogy, una colonna per transizione) ─────────
_sw_csv = os.path.join(_swdiag_loss_dir, 'stage_switch.csv')
if os.path.isfile(_sw_csv):
    df_sw = pd.read_csv(_sw_csv)

    # calcola R_lb se non già presente
    eps = 1e-12
    beta1 = df_sw['beta1'].to_numpy()
    beta2 = df_sw['beta2'].to_numpy()
    g     = np.abs(df_sw['gnew_median'].to_numpy())
    delta = df_sw['delta_median'].to_numpy()
    sigma = df_sw['sigma_median'].to_numpy()
    R_lb  = (np.sqrt(beta2) * sigma * (1.0 - beta1)) / (
             np.sqrt(1.0 - beta2) * ((1.0 - beta1) * g + beta1 * delta) + eps)

    # raccogli transizioni in ordine naturale
    pairs = []
    for (sfrom, sto), idx in df_sw.groupby(['stage_from', 'stage_to']).groups.items():
        pairs.append((sfrom, sto, list(idx)))
    pairs.sort(key=lambda r: (_stage_order(r[0]), _stage_order(r[1])))
    pairs = pairs[:2]   # tipicamente Stage1→2 e Stage2→3

    if pairs:
        fig_rlb, axes_rlb = plt.subplots(1, len(pairs),
                                          figsize=(6 * len(pairs), 4),
                                          sharey=True)
        if len(pairs) == 1:
            axes_rlb = [axes_rlb]
        for ax, (sfrom, sto, idx) in zip(axes_rlb, pairs):
            vals = R_lb[np.array(idx)]
            ax.semilogy(np.arange(len(vals)), vals, 'o',
                        markersize=4, alpha=0.8, color='#1f4e79')
            ax.axhline(1.0, ls='--', lw=1.2, color='tab:red', alpha=0.8, label=r'$R_{\mathrm{lb}}=1$')
            med = float(np.median(vals))
            ax.axhline(med, ls=':', lw=1.2, color='gray', alpha=0.7,
                       label=f'median = {med:.2f}')
            ax.set_title(f'{_PRETTY.get(sfrom, sfrom)} → {_PRETTY.get(sto, sto)}',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Layer index', fontsize=13)
            ax.tick_params(labelsize=11)
            ax.legend(fontsize=11, framealpha=0.9)
            ax.grid(True, which='both', alpha=0.25)
        axes_rlb[0].set_ylabel(r'$R_{\mathrm{lb}}$', fontsize=14)
        fig_rlb.tight_layout()
        _out = os.path.join(OPT_DIR, 'Rlb_per_layer.png')
        fig_rlb.savefig(_out, dpi=200)
        plt.close(fig_rlb)
        print(f'→ Saved: {_out}')

    # ── C: Check3 log–log (beta2*vhat_{t-1} vs (1-beta2)*|g_t|^2) ───────────
    fig_c3, ax_c3 = plt.subplots(figsize=(6, 6))
    PALETTE_C3 = ['#1f4e79', '#b33c2e', '#5a2a83', '#1b7d4a']
    for i, (sfrom, sto, idx) in enumerate(pairs):
        sub = df_sw.iloc[idx]
        b2  = sub['beta2'].to_numpy()
        sig = sub['sigma_median'].to_numpy()
        gabs = np.abs(sub['gnew_median'].to_numpy())
        lhs3 = np.clip((1.0 - b2) * gabs**2, eps, None)
        rhs3 = np.clip(b2 * sig**2,           eps, None)
        lbl  = f'{_PRETTY.get(sfrom, sfrom)} → {_PRETTY.get(sto, sto)}'
        ax_c3.loglog(rhs3, lhs3, '.', markersize=6, alpha=0.85,
                     color=PALETTE_C3[i % len(PALETTE_C3)], label=lbl)

    # diagonale y=x
    _lims = [min(ax_c3.get_xlim()[0], ax_c3.get_ylim()[0]),
             max(ax_c3.get_xlim()[1], ax_c3.get_ylim()[1])]
    ax_c3.plot(_lims, _lims, 'k--', lw=0.9, alpha=0.5, label='y = x')
    ax_c3.set_xlabel(r'$\beta_2\,\widehat{v}_{t-1}$', fontsize=14)
    ax_c3.set_ylabel(r'$(1-\beta_2)\,\|g_t\|^2$', fontsize=14)
    ax_c3.tick_params(labelsize=12)
    ax_c3.legend(fontsize=12, framealpha=0.9)
    ax_c3.grid(True, which='both', alpha=0.25)
    fig_c3.tight_layout()
    _out = os.path.join(OPT_DIR, 'check3_loglog.png')
    fig_c3.savefig(_out, dpi=200)
    plt.close(fig_c3)
    print(f'→ Saved: {_out}')

else:
    print(f'⚠  stage_switch.csv not found in {_swdiag_loss_dir} — optimizer switch plots skipped.')

print('\nDone.')
