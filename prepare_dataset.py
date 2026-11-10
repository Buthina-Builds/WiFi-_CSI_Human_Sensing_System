
import csv
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"D:\New folder (2)\csi_dataset")
OUTPUT_FILE = DATA_DIR / "csi_features_dataset.csv"

WINDOW_SIZE = 97  # approximately one second

# Remove known constant/mostly-zero subcarriers
VALID_SUBCARRIERS = [
    i for i in range(64)
    if i not in [0, 1] and not 28 <= i <= 36
]

def extract_features(rows):
    csi_matrix = np.array([r["csi"] for r in rows])
    rssi_values = np.array([r["rssi"] for r in rows])

    features = {}

    # Statistics for each valid subcarrier
    for index in VALID_SUBCARRIERS:
        values = csi_matrix[:, index]
        features[f"sc_{index}_mean"] = np.mean(values)
        features[f"sc_{index}_std"] = np.std(values)

    # Overall CSI statistics
    row_means = np.mean(csi_matrix, axis=1)

    features["csi_mean"] = np.mean(csi_matrix)
    features["csi_std"] = np.std(csi_matrix)
    features["csi_median"] = np.median(csi_matrix)
    features["csi_p90"] = np.percentile(csi_matrix, 90)

    # Temporal variation
    features["temporal_change"] = (
        np.mean(np.abs(np.diff(row_means)))
        if len(row_means) > 1 else 0
    )

    features["window_variation"] = np.std(row_means)

    # RSSI statistics
    features["rssi_mean"] = np.mean(rssi_values)
    features["rssi_std"] = np.std(rssi_values)

    return features


all_samples = []
files = sorted(DATA_DIR.glob("*20261109*.csv"))

print(f"Found {len(files)} recordings")

for file_path in files:
    filename = file_path.name

    if filename.startswith("NO_PERSON"):
        label = "NO_PERSON"
    elif filename.startswith("PERSON_STATIONARY"):
        label = "STATIONARY"
    elif filename.startswith("PERSON_MOVING"):
        label = "MOVING"
    else:
        continue

    rows = []

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                raw = json.loads(row["data"])

                if len(raw) != 128:
                    continue

                iq = np.array(raw, dtype=float).reshape(64, 2)
                magnitudes = np.sqrt(
                    iq[:, 0] ** 2 + iq[:, 1] ** 2
                )

                rows.append({
                    "csi": magnitudes,
                    "rssi": float(row["rssi"])
                })

            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue

    # Create non-overlapping one-second windows
    window_count = len(rows) // WINDOW_SIZE

    for w in range(window_count):
        start = w * WINDOW_SIZE
        window = rows[start:start + WINDOW_SIZE]

        features = extract_features(window)
        features["label"] = label
        features["recording"] = filename

        all_samples.append(features)

    print(
        f"{filename}: {len(rows)} valid rows, "
        f"{window_count} windows"
    )


if not all_samples:
    raise RuntimeError("No valid windows were created!")

fieldnames = list(all_samples[0].keys())

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_samples)

print("\n================================")
print("DATASET PREPARATION COMPLETE")
print("================================")
print(f"Total windows: {len(all_samples)}")
print(f"Output: {OUTPUT_FILE}")

for label in ["NO_PERSON", "STATIONARY", "MOVING"]:
    count = sum(s["label"] == label for s in all_samples)
    print(f"{label}: {count} windows")