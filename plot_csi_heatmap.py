import csv
import glob
import os
import ast

import numpy as np
import matplotlib.pyplot as plt

folder = r"D:\New folder (2)\csi_dataset"

files = {
    "NO PERSON": "NO_PERSON_20261109_001735.csv",
    "MOVING": "PERSON_MOVING_20261109_002536.csv",
    "STATIONARY": "PERSON_STATIONARY_20261109_002241.csv"
}

all_values = []

for filename in files.values():
    path = os.path.join(folder, filename)

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            values = ast.literal_eval(row["data"])
            iq = np.array(values, dtype=float).reshape(-1, 2)
            magnitude = np.sqrt(iq[:, 0]**2 + iq[:, 1]**2)
            all_values.append(magnitude)

vmin = min(x.min() for x in all_values)
vmax = max(x.max() for x in all_values)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (label, filename) in zip(axes, files.items()):
    path = os.path.join(folder, filename)

    csi_data = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                values = ast.literal_eval(row["data"])
                iq = np.array(values, dtype=float).reshape(-1, 2)

                magnitude = np.sqrt(
                    iq[:, 0]**2 + iq[:, 1]**2
                )

                csi_data.append(magnitude)

            except Exception:
                continue

    matrix = np.array(csi_data)

    # Normalize color scale across all three conditions later
    image = ax.imshow(
    matrix.T,
    aspect="auto",
    origin="lower",
    cmap="jet",
    vmin=vmin,
    vmax=vmax)
    
    ax.set_title(label)
    ax.set_xlabel("Time (CSI samples)")
    ax.set_ylabel("Subcarrier")

fig.colorbar(image, ax=axes, label="CSI Magnitude")
fig.suptitle("Real Wi-Fi CSI Heatmaps", fontsize=16)
plt.tight_layout()

output = os.path.join(folder, "CSI_Heatmaps.png")
plt.savefig(output, dpi=200)
plt.show()

print("\nHEATMAP CHECK")

for label, filename in files.items():
    path = os.path.join(folder, filename)
    values_list = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            values = ast.literal_eval(row["data"])
            iq = np.array(values, dtype=float).reshape(-1, 2)
            magnitude = np.sqrt(iq[:, 0]**2 + iq[:, 1]**2)
            values_list.append(magnitude)

    matrix = np.array(values_list)

    print(f"\n{label}")
    print("Shape:", matrix.shape)
    print("Min:", round(matrix.min(), 2))
    print("Max:", round(matrix.max(), 2))
    print("Mean:", round(matrix.mean(), 2))
    print("Std:", round(matrix.std(), 2))

print("Heatmap saved to:", output)

