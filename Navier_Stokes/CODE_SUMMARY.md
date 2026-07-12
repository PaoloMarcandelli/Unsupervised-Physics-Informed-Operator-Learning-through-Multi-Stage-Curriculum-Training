# Code Summary — Navier-Stokes, Kolmogorov & Kelvin-Helmholtz (2D PhIS-FNO)

Questo documento riassume la struttura e il flusso completo del codice per gli esperimenti 2D,
mettendoli in relazione con la metodologia descritta nel paper.

---

## 1. Struttura delle directory

```
Navier_Stokes/
├── train_model.py              # Script di training unificato (punto di ingresso)
├── plot_loss.py                # Visualizzazione loss unificata
├── run_experiments.sh          # Script bash per lanciare batch di esperimenti
├── shared/                     # Moduli condivisi tra tutti gli esperimenti
│   ├── models.py               # Tutte le architetture (PhIS-FNO, PINO, UNet, ResNet, TFNet, DeepONet)
│   ├── spline_models.py        # Kernel Hermite spline + interpolate_states
│   ├── setups.py               # Classe Dataset + maschera di boundary
│   ├── operators.py            # Differenziazione automatica (grad, rot, div, laplace)
│   ├── utilities3.py           # LpLoss, LpLossSafe, MatReader, normalizzatori
│   ├── unet_parts.py           # Blocchi UNet (DoubleConv, Down, Up, OutConv)
│   ├── ns_2d.py                # Solver pseudo-spettrale NS per generazione dataset
│   ├── random_fields.py        # Campionamento Gaussian Random Fields
│   ├── orthogonal.py           # (Sperimentale) Spectral conv con mappa esponenziale ortogonale
│   └── resolution.py           # Analisi zero-shot super-resolution
├── navier-stokes-vorticity/    # Dati e output per NS con ν = 1e-3 e ν = 1e-4
│   ├── ns_data_{nu}.mat
│   ├── model/
│   └── loss/
├── kolmogorov/                 # Dati e output per Kolmogorov flow (ν = 2e-3)
│   ├── ns_data_Kolmogorov.mat
│   ├── model/
│   └── loss/
└── kelvin-helmholtz/           # Dati e output per Kelvin-Helmholtz instability
    ├── simulation_dataset.mat  # Generato da finitevolume.py
    ├── finitevolume.py         # Generatore dataset KH (schema Muscl-Hancock)
    ├── visualize.py            # Rollout autoregressivo + GIF 3-panel
    ├── model/
    └── loss/
```

---

## 2. Script di training — `train_model.py` (unificato)

Un unico script sostituisce i precedenti script separati per architettura. Va lanciato dalla directory `Navier_Stokes/`.

```bash
python train_model.py \
    --exp_dir  navier-stokes-vorticity | kolmogorov | kelvin-helmholtz \
    --model    phisfno | pino | unet | resnet | tfnet \
    --training multistage | multistage_nr | singlestage | supervised \
    [--nu 1e-3] [--offset 40] [--epochs 100] ...
```

Il flag `--exp_dir` viene parsato **prima** degli altri import (`parse_known_args`): lo script esegue
`os.chdir(exp_dir)` e aggiunge `shared/` al `sys.path`, così i moduli vengono importati correttamente.

---

## 3. Architetture (shared/models.py)

### 3.1 Net2d — PhIS-FNO (modello principale)

Wrapper di `SimpleBlock2d`.

```
Input  (B, S, S, 12)        # [x, y, ω_{t-T_in+1}, ..., ω_t]
  → fc0: Linear(12 → width)
  → permute → (B, width, S, S)
  → 4× [SpectralConv2d_fast + Conv1d(skip) + BatchNorm2d + ReLU]
  → permute → (B, S, S, width)
  → fc1: Linear(width → 128) + ReLU
  → fc2: Linear(128 → 9)     # 9 coefficienti spline (ordine 2,2: (2+1)² = 9)
  → permute → (B, 9, S, S)
  → crop [1:, 1:]             # allineamento griglia
Output (B, 9, S, S)           # coefficienti spline
```

`SpectralConv2d_fast`: rfft2 → moltiplicazione con pesi complessi learnable sui modi
`[0:m, 0:m]` e `[-m:, 0:m]` → irfft2.

### 3.2 PINONet2d — PINO

Identica struttura FNO di `SimpleBlock2d`, ma `fc2: Linear(128 → 1)` — predice direttamente ω,
senza decoder spline. Le derivate per il residuo fisico vengono calcolate in spazio spettrale via
`spectral_ops` nel training loop.

### 3.3 fluid_model — UNet con spline

Encoder-decoder UNet standard (4 Down + 4 Up con skip connections bilinear).
- Output: 9 canali spline con `tanh` + normalizzazione a media zero.
- Crop finale `[..., 1:, 1:]` per allineamento griglia.
- Usa `LpLossSafe` per la metrica di eval (evita NaN da batch outlier).

### 3.4 ResNet18_NS — ResNet baseline

ResNet-18 stile per predizione one-step di vorticità.
- Stem: conv7×7 + GroupNorm + GELU.
- 8 `ResNetBasicBlock2D` (conv3×3 × 2 + residual).
- Head: conv3×3 + GN + GELU + conv1×1.
- Residual connection opzionale: `y = y + x[..., -1:]` (ultimo frame input).
- Input: `(B,S,S, 2+T_in)`, Output: `(B,S,S,1)` — ω diretto, no spline.

### 3.5 TFNet_NS — TF-Net baseline

Architettura temporal-fusion per flussi turbolenti.
- `TFTemporalBlock`: Conv3D sullo stack temporale di ω → aggrega la dimensione tempo.
- `TFSpatialBlock`: tre branch paralleli (kernel 3×3, 5×5, dilatato 3×3) sulla concatenazione `[x,y,ω-stack]`.
- `TFFusionBlock` × depth: residual conv3×3.
- Head: conv3×3 + GN + GELU + conv1×1.
- Residual connection opzionale sull'ultimo frame.
- Input: `(B,S,S, 2+T_in)`, Output: `(B,S,S,1)` — ω diretto, no spline.

### 3.6 DeepONet2D_NS_Vorticity — (sperimentale, non in train_model.py)

Architettura DeepONet per NS one-step.
- Branch: `BranchCNN2D` — encoder CNN sul T_in stack di ω → vettore latente `(B, latent)`.
- Trunk: `TrunkMLP` con `FourierFeatures` per encoding posizionale — per-pixel su (x,y) → `(B,S,S,latent)`.
- Output: dot-product branch·trunk + bias → `(B,S,S,1)`.
- Supporta conditioning opzionale su dt, nu, scalari extra.

---

## 4. Decodifica spline (shared/spline_models.py)

La funzione centrale è `interpolate_2d_velocity`:

1. Costruisce i **kernel Hermite 2D** (prodotto tensore di basi 1D `p2_1`, `p2_2`) per un dato offset nella cella.
2. I kernel coprono: ψ (stream function), v = rot(ψ), ∇v, ω = ∂v_y/∂x − ∂v_x/∂y, ∇ω, Δω.
3. Applica `F.conv2d` con **padding circolare** dei coefficienti `(B,9,S,S)` con i kernel precomputati.
4. I kernel sono **bufferizzati su disco** (`Logger/buffers/buffers.dic`) per evitare ricalcoli.

`interpolate_states` (wrapper pubblico) restituisce `(ω, v, ∇ω, Δω)` in formato `(B, S, S, C)`.

**Vantaggio architetturale chiave (paper §2.4.3):** la rete predice coefficienti spline, non direttamente
il campo. Le derivate si ottengono applicando i kernel della derivata degli stessi spline agli stessi
coefficienti — questo fornisce derivate analiticamente consistenti con la predizione, a differenza di
PINO/ResNet/TFNet che usano derivate spettrali a posteriori sul campo predetto.

---

## 5. Dataset e preprocessing

**Campi caricati dal file .mat:**
```python
train_a = u[:ntrain, ::sub, ::sub, offset : offset+T_in]      # (N, S, S, T_in) — input
train_u = u[:ntrain, ::sub, ::sub, offset+T_in : offset+T+T_in]  # (N, S, S, T) — target
```

**Input finale al modello (griglia concatenata):**
```python
train_a = [x_grid, y_grid, ω_{offset:offset+T_in}]  # (B, S, S, 2+T_in)
```

- Griglia spaziale: `linspace(0,1,S)`, S=64 (subsampling 4× da 256).
- **Offset temporale:** 40 per ν=1e-3, 20 per ν=1e-4, 30 per Kolmogorov (flusso già turbolento).
- Training **autoregressivo**: per ogni step t, il modello riceve gli ultimi T_in frame e predice il
  successivo, aggiornando il buffer di input.

**Per Kelvin-Helmholtz:**
- La variabile `u` nel .mat contiene la **densità ρ** (non vorticità).
- Se disponibili, vengono caricati anche `vx` e `vy` (velocità ground-truth) per il residuo fisico.
- dt = 0.1 (KH, tEnd=2.0 / 20 step); dt = 0.25 (NS/Kolmogorov).

**Maschera di boundary (setups.py, pad=1):**
```python
v_mask[:, :1, :] = 1   # bordo superiore
v_mask[:, -1:, :] = 1  # bordo inferiore
v_mask[:, :, :1] = 1   # bordo sinistro
v_mask[:, :, -1:] = 1  # bordo destro
```
Bordo di 1 pixel su S×S. L'interior mask è `1 - v_mask`.

---

## 6. Flusso di training (train_model.py)

### 6.1 Fasi del curriculum

| Esperimento | Modello | Fasi | Struttura |
|---|---|---|---|
| Navier-Stokes | tutti | 5 | 2 boundary-only + 3 full-loss progressivi |
| Kolmogorov | tutti | 3 | 1 boundary-only + 2 full-loss progressivi |
| Kelvin-Helmholtz | spline (phisfno/unet) | 3 | λ_res=0 in fase 1, poi progressivo |
| Kelvin-Helmholtz | non-spline (pino/resnet/tfnet) | 3 | λ_res=0.2 già in fase 1 |

**Pesi per NS:**
```
boundary_only_1: λ_sup=1.0, λ_res=0.0
boundary_only_2: λ_sup=1.0, λ_res=0.0
full_loss_1:     λ_sup=0.8, λ_res=0.5
full_loss_2:     λ_sup=0.5, λ_res=1.0
full_loss_3:     λ_sup=0.2, λ_res=1.5
```

**Pesi per Kolmogorov:**
```
boundary_only_1: λ_sup=0.5,  λ_res=0.1
full_loss_1:     λ_sup=0.4,  λ_res=0.25
full_loss_2:     λ_sup=0.25, λ_res=0.5
```

**Pesi per Kelvin-Helmholtz:**
```
boundary_only_1: λ_sup=1.0, λ_res=0.0  (spline) | λ_res=0.2 (non-spline)
full_loss_1:     λ_sup=0.5, λ_res=0.5
full_loss_2:     λ_sup=0.2, λ_res=1.0
```

**Reset Adam:** a ogni fase un nuovo `torch.optim.Adam` viene istanziato (multistage).
**No-reset (multistage_nr):** stesso ottimizzatore riusato tra le fasi.

### 6.2 Loss per step autoregressivo

```
loss_total += λ_sup    * loss_sup_step   # Lp boundary
           +  λ_res    * loss_res_step   # MSE residuo fisico su interior
           +  λ_smooth * smooth_loss     # solo modelli spline
           +  λ_tv     * tv_loss         # solo modelli spline
           +  λ_mean   * mean_v_loss     # solo modelli spline
```

**Regolarizzatori spline** (phisfno, unet — non applicati a pino/resnet/tfnet):
- `smooth`: L2 dei gradienti finiti di ω in x e y
- `tv`: L1 dei gradienti (Total Variation)
- `mean_v`: norma L2 della media spaziale della velocità (vincolo gauge)

Con λ_smooth=0.1, λ_tv=0.01, λ_mean=0.01.

### 6.3 Residuo fisico

**NS e Kolmogorov** (formulazione vorticità, tutti i modelli):
```
dω/dt = (ω_pred - ω_{t-1}) / dt
r = dω/dt + u·∇ω − ν·Δω − f(x,y)
```

**Kelvin-Helmholtz con velocità ground-truth** (percorso prioritario se vx/vy nel .mat):
```
r = ∂ρ/∂t + ∇·(ρ·v_gt)   # continuità comprimibile in forma conservativa
```

**Kelvin-Helmholtz con velocità spline** (phisfno/unet senza gt velocity):
```
r = ∂ρ/∂t + ∇·(ρ·v_spline)
```

**Kelvin-Helmholtz fallback** (pino/resnet/tfnet senza gt velocity):
```
r = ∂ρ/∂t + u_BiotSavart·∇ρ − ν·Δρ − f   # applica eq. NS vorticity al campo ρ
```
Nota: questo fallback applica un'equazione fisicamente scorretta per KH comprimibile.
È raccomandato generare sempre il dataset con vx/vy (flag `--include_velocity` in finitevolume.py).

**Derivate spettrali** (`spectral_ops`): FFT → Poisson solver per ψ → u=∂ψ/∂y, v=−∂ψ/∂x → ∇ω, Δω.
De-aliasing con regola 2/3 sul termine convettivo. Il dominio è assunto periodico (toro).

### 6.4 Boundary loss — confronto modelli

| Modello | Calcolo | Pixel usati | Normalizzazione |
|---|---|---|---|
| phisfno, unet | `lp_on_mask(loss_bd, ω, y, bm_bool)` | Solo i ~4·S pixel del bordo | `||y_border||_p` |
| pino, resnet, tfnet | `loss_bd((im·bm).reshape(...), (y·bm).reshape(...))` | S² pixel, interni azzerati | `||y·bm||_p` (≡ `||y_border||_p`) |

Le due implementazioni sono **matematicamente equivalenti** per la relative Lp loss: i pixel interni
sono zero sia nel numeratore che nel denominatore, quindi non cambiano il rapporto.
Il gradiente che arriva al modello è identico in entrambi i casi.

### 6.5 Forzante

| Esperimento | f(x,y) |
|---|---|
| Navier-Stokes | `0.1·(sin(2π(x+y)) + cos(2π(x+y)))` |
| Kolmogorov | `−2·cos(4πy)` (n_force=2 in ns_2d.py) |
| Kelvin-Helmholtz | `0` (Euler, no forzante) |

### 6.6 DataLoader shuffle

- `pino`, `resnet`, `tfnet`: `shuffle=True` (Generator seedato per riproducibilità)
- `phisfno`, `unet`: `shuffle=False`

---

## 7. Dataset Kelvin-Helmholtz (kelvin-helmholtz/finitevolume.py)

Schema **Muscl-Hancock** (finite volume, 2° ordine in spazio e tempo).

**Condizioni iniziali:**
- Strato di shear: ρ=2, vx=−0.5 dentro (|y−0.5|<0.25); ρ=1, vx=+0.5 fuori
- Perturbazione sinusoidale in vy + rumore casuale seedato

**Parametri:**
- Griglia: 128×128, γ=5/3
- dt adattivo via CFL, tEnd=2.0, 20 snapshot di output
- Flusso numerico: Rusanov (HLL-like con velocità d'onda massima)

**Output .mat:** `u` (ρ), `vx`, `vy` — shape `(N, 128, 128, T_out)`

---

## 8. Differenze tra esperimenti

| Aspetto | Navier-Stokes | Kolmogorov | Kelvin-Helmholtz |
|---|---|---|---|
| Dataset | `ns_data_{nu}.mat` | `ns_data_Kolmogorov.mat` | `simulation_dataset.mat` |
| Variabile fisica | vorticità ω | vorticità ω | densità ρ |
| Viscosità | 1e-3, 1e-4 | 2e-3 | 0 (Euler) |
| Offset T_in | 40 (ν=1e-3), 20 (ν=1e-4) | 30 | configurabile |
| Forzante | trigonometrico | sinusoidale in y | nessuno |
| Fasi MS | 5 | 3 | 3 |
| dt snapshot | 0.25 | 0.25 | 0.1 |
| Generatore | ns_2d.py (pseudo-spettrale) | ns_2d.py | finitevolume.py (FV Muscl-Hancock) |

---

## 9. Parametri CLI principali

| Argomento | Default | Note |
|---|---|---|
| `--exp_dir` | (required) | `navier-stokes-vorticity` \| `kolmogorov` \| `kelvin-helmholtz` |
| `--model` | (required) | `phisfno` \| `pino` \| `unet` \| `resnet` \| `tfnet` |
| `--training` | (required) | `multistage` \| `multistage_nr` \| `singlestage` \| `supervised` |
| `--ntrain` | 16 | campioni di training |
| `--ntest` | 2 | campioni di test |
| `--modes` | 12 | modi di Fourier (m nel paper) |
| `--width` | 20 | canali nascosti PhIS-FNO/PINO |
| `--resnet_width` | 64 | canali ResNet |
| `--tfnet_width` | 64 | canali TFNet |
| `--tfnet_depth` | 4 | numero fusion block TFNet |
| `--bsize` | 2 | batch_size = ntrain // bsize |
| `--epochs` | 100 | epoch per fase (multistage) |
| `--ss_epochs` | 150 | epoch singlestage |
| `--sup_epochs` | 150 | epoch supervised |
| `--learning_rate` | 2e-3 | Adam lr |
| `--scheduler_step` | 20 | decay ogni N epoch |
| `--scheduler_gamma` | 0.5 | fattore di decay |
| `--S` | 64 | risoluzione spaziale |
| `--T_in` | 10 | lunghezza sequenza input |
| `--T` | 10 | orizzonte di previsione |
| `--offset` | 40 | offset temporale |
| `--nu` | 1e-3 | viscosità cinematica |
| `--seed` | 0 | seme random |
| `--orders_v` | [2,2] | ordine spline 2D → (2+1)²=9 coeff |
| `--rf` | 8 | fattore super-risoluzione |

### Naming dei path di output

```
model/{model}_{tag}_{ntrain}_Adam_ep{ep}_visc{nu}_m{modes}_w{width}_T{T}
loss/{model}_{tag}_{ntrain}_Adam_ep{ep}_visc{nu}_m{modes}_w{width}_T{T}/loss_{phase_name}.csv
```

Con `tag` ∈ {ms, ms_nr, ss, sup} e `ep` = valore effettivo per quella strategia.

---

## 10. Visualizzazione

### plot_loss.py

Genera grafici delle loss per modello/strategia con rolling mean e bande di errore (scala log).
Funzioni principali:
- `infer_phase_names`: rileva automaticamente se 3 o 5 fasi (controlla `boundary_only_2`)
- `load_test_full_series`: carica `test_full` da tutti i CSV di fase
- `plot_model_family`: MS vs MS-nr vs SS vs Sup con boundaries di fase verticali
- `rolling_stats_log`: media mobile in spazio logaritmico

### kelvin-helmholtz/visualize.py

Rollout autoregressivo su un campione di test → GIF 3-panel (predizione | GT | errore).
- `predict`: rollout con decodifica spline opzionale
- `make_gif`: animazione con `matplotlib.animation.FuncAnimation`
- `save_snapshot`: griglia 3×3 per 3 istanti temporali

---

## 11. Connessioni con il paper

| Concetto paper | Implementazione |
|---|---|
| Eq. (7): loss pesata | `λ_sup * loss_sup + λ_res * loss_res` in train_model.py |
| Eq. (9): loss NS spazio-tempo | residuo `dω/dt + conv − ν·Δω − f` in `spectral_ops` |
| Eq. (13-15): reset Adam | `optimizer = torch.optim.Adam(...)` per ogni fase (multistage) |
| Eq. (16-17): no-reset | stesso optimizer nei file multistage_nr |
| §2.4.1: kernel Hermite spline | `spline_models.py`: `p2_1, p2_2, p_multidim` |
| §2.4.3: PhIS-FNO predice coeff | fc2 → 9 output, poi `interpolate_states` |
| §2.4.2: propagazione globale boundary | `SpectralConv2d_fast`: convoluzione globale in Fourier |
| Tab A.2: pesi per fase | lista `phases` in `build_phases()` |
| Tab A.3: iperparametri | default in `get_params()` |
| Sec 3.3: instabilità PINO su griglia fine | PINO usa `spectral_ops`; PhIS-FNO usa kernel spline |
| Dealiasing 2/3 | `make_dealias_mask(S)` applicata al termine convettivo |
