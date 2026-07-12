# Generate a log-scale plot for relative errors at different resolutions
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os

# Data
x = np.array([32, 64, 128, 256])
y_ns_1e3 = np.array([0.071212694, 0.064028993, 0.063387126, 0.063377909])  # Navier–Stokes ν=1e-3
# y_ns_1e4 = np.array([0.12886434,  0.12477279,  0.124412149, 0.1242973])    # Navier–Stokes ν=1e-4
# y_kolm   = np.array([0.138679,    0.130449,    0.130110,    0.130218])     # Kolmogorov flow
y_ns_1e3_unet = np.array([0.148960, 0.107557, 0.556503, 1.298309]) 
plt.figure(figsize=(7,4.5))

# Curves
plt.plot(x, y_ns_1e3, marker='o', label="PhIS-FNO")
plt.plot(x, y_ns_1e3_unet, marker='o', label="UNet")
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
out_path = os.path.join(out_dir, "resolution_multi.png")

plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"✔️ Salvato: {out_path}")
