# Generate a log-scale plot for relative errors at different resolutions
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os

matplotlib.rcParams.update({
    'font.size': 16,           # dimensione base
    'axes.titlesize': 20,      # titoli dei subplot
    'axes.labelsize': 16,      # etichette assi
    'xtick.labelsize': 16,     # tick x
    'ytick.labelsize': 16,     # tick y
    'legend.fontsize': 16,     # legende
    'figure.titlesize': 20,    # titolo generale
})


MAGENTA = "#FF2D95"  # neon magenta / hot pink
CYAN    = "#00E5FF"  # neon cyan
ORANGE  = "#FF6D00" 
# Data
x = np.array([32, 64, 128, 256])
y_ns_1e3 = np.array([0.071212694, 0.064028993, 0.063387126, 0.063377909])  # Navier–Stokes ν=1e-3
y_ns_1e4 = np.array([0.10911713540554047,  0.10443119704723358,  0.10414823144674301, 0.104122593998909])    # Navier–Stokes ν=1e-4
# y_kolm   = np.array([0.138679,    0.130449,    0.130110,    0.130218])     # Kolmogorov flow
y_ns_1e3_unet = np.array([0.148960, 0.107557, 0.556503, 1.298309]) 
y_ns_1e4_unet = np.array([0.22358405590057373, 0.16522450745105743,  0.2806819677352905, 0.23793534934520721])

y_ns_1e3_pino = np.array([0.06038922443985939,  0.06021029129624367, 0.060137007385492325, 0.06006674841046333]) 
y_ns_1e4_pino = np.array([0.09154012799263,0.09139326959848404,  0.0912887305021286, 0.0912538692355156])

plt.figure(figsize=(7,4.5))

# Curves
plt.plot(x, y_ns_1e4, marker='o',color=MAGENTA, label="PhIS-FNO")
plt.plot(x, y_ns_1e4_unet, marker='o',color=CYAN, label="UNet")
plt.plot(x, y_ns_1e4_pino, marker='o',color=ORANGE, label="PINO")
# plt.plot(x, y_ns_1e4, marker='s', label=r"Navier–Stokes  $\nu=10^{-4}$")
# plt.plot(x, y_kolm,   marker='^', label="Kolmogorov flow")

# Axes and scales
plt.yscale("log")
plt.xlabel("Resolution (grid size)")
plt.ylabel("Relative error")
plt.ylim(1e-2, 1e1)

# Ticks: show all x values explicitly
plt.xticks(x, labels=[str(v) for v in x])

plt.title(r"Navier Stokes Equations $\nu =1e-3$")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.tight_layout()

out_dir = "image"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "resolution_multi_nu1e_4.png")

plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"✔️ Salvato: {out_path}")
