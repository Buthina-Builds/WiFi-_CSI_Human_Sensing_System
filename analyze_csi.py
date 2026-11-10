import csv, glob, os, ast
import numpy as np

folder = r"D:\New folder (2)\csi_dataset"

for path in sorted(glob.glob(os.path.join(folder, "*20261109*.csv"))):
    name = os.path.basename(path)

    if "160658" in name:
        continue

    rssi_values = []
    csi_means = []
    csi_stds = []
    lengths = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                rssi_values.append(float(row["rssi"]))
                values = ast.literal_eval(row["data"])

                if len(values) < 4 or len(values) % 2:
                    continue

                values = np.array(values, dtype=float)
                iq = values.reshape(-1, 2)
                magnitudes = np.sqrt(iq[:, 0]**2 + iq[:, 1]**2)

                csi_means.append(np.mean(magnitudes))
                csi_stds.append(np.std(magnitudes))
                lengths.append(len(values))

            except Exception:
                continue

    print("\n" + "=" * 55)
    print("FILE:", name)
    print("Valid analyzed rows:", len(csi_means))

    if csi_means:
        print("Average RSSI:", round(np.mean(rssi_values), 2), "dBm")
        print("Average CSI magnitude:", round(np.mean(csi_means), 2))
        print("CSI magnitude variation:", round(np.mean(csi_stds), 2))
        print("CSI values per row:", sorted(set(lengths)))


print("\nTEMPORAL CSI ANALYSIS")

for path in sorted(glob.glob(os.path.join(folder, "*20261109*.csv"))):

    name = os.path.basename(path)

    if "160658" in name:
        continue

    samples = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                values = ast.literal_eval(row["data"])
                iq = np.array(values, dtype=float).reshape(-1, 2)

                magnitude = np.sqrt(
                    iq[:, 0]**2 + iq[:, 1]**2
                )

                samples.append(magnitude)

            except Exception:
                continue

    matrix = np.array(samples)

    # Change between consecutive CSI samples
    differences = np.diff(matrix, axis=0)

    temporal_change = np.mean(np.abs(differences))

    print("\n", name)
    print("Average temporal change:", round(temporal_change, 4))

print("\nWINDOW-BASED CSI ANALYSIS")

for path in sorted(glob.glob(os.path.join(folder, "*20261109*.csv"))):

    name = os.path.basename(path)

    if "160658" in name:
        continue

    samples = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                values = ast.literal_eval(row["data"])
                iq = np.array(values, dtype=float).reshape(-1, 2)

                magnitude = np.sqrt(
                    iq[:, 0]**2 + iq[:, 1]**2
                )

                samples.append(magnitude)

            except Exception:
                continue

    matrix = np.array(samples)

    window_size = 97
    window_changes = []

    for start in range(0, len(matrix) - window_size, window_size):

        window = matrix[start:start + window_size]

        change = np.mean(np.abs(np.diff(window, axis=0)))

        window_changes.append(change)

    window_changes = np.array(window_changes)

    print("\n", name)
    print("Number of windows:", len(window_changes))
    print("Median:", round(np.median(window_changes), 4))
    print("90th percentile:", round(np.percentile(window_changes, 90), 4))
    print("Minimum:", round(np.min(window_changes), 4))
    print("Maximum:", round(np.max(window_changes), 4))

print("\n" + "=" * 55)
print("SUBCARRIER ANALYSIS")

conditions = {
    "NO PERSON": "NO_PERSON_20261109_014900.csv",
    "STATIONARY": "PERSON_STATIONARY_20261109_015120.csv",
    "MOVING": "PERSON_MOVING_20261109_015337.csv"
}

subcarrier_data = {}

for label, filename in conditions.items():
    path = os.path.join(folder, filename)
    samples = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                values = ast.literal_eval(row["data"])
                iq = np.array(values, dtype=float).reshape(-1, 2)

                magnitude = np.sqrt(
                    iq[:, 0]**2 + iq[:, 1]**2
                )

                if len(magnitude) == 64:
                    samples.append(magnitude)

            except Exception:
                continue

    matrix = np.array(samples)
    subcarrier_data[label] = matrix

    print(f"\n{label}")
    print("Shape:", matrix.shape)
    print("Mean magnitude per subcarrier:")
    print(np.round(np.mean(matrix, axis=0), 2))

    print("Variation per subcarrier:")
    print(np.round(np.std(matrix, axis=0), 2))


print("\n" + "=" * 55)
print("TOP DISCRIMINATIVE SUBCARRIERS")

labels = list(subcarrier_data.keys())

means = {
    label: np.mean(subcarrier_data[label], axis=0)
    for label in labels
}

stds = {
    label: np.std(subcarrier_data[label], axis=0)
    for label in labels
}

scores = []

for i in range(64):

    values = [means[label][i] for label in labels]

    if max(values) == 0:
        continue

    if np.std(values) == 0:
        continue

    score = np.std(values)
    scores.append((i, score))

scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop 15 subcarriers:")

for index, score in scores[:15]:
    print(
        f"Subcarrier {index:2d} | "
        f"Score: {score:.4f} | "
        f"No Person: {means['NO PERSON'][index]:.2f} | "
        f"Moving: {means['MOVING'][index]:.2f} | "
        f"Stationary: {means['STATIONARY'][index]:.2f}"
    )

print("\n" + "=" * 55)
print("SELECTED SUBCARRIER WINDOW ANALYSIS")

selected = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
window_size = 97

for label, filename in conditions.items():

    path = os.path.join(folder, filename)
    samples = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                values = ast.literal_eval(row["data"])
                iq = np.array(values, dtype=float).reshape(-1, 2)
                magnitude = np.sqrt(
                    iq[:, 0]**2 + iq[:, 1]**2
                )

                samples.append(magnitude[selected])

            except Exception:
                continue

    matrix = np.array(samples)
    window_means = []

    for start in range(0, len(matrix) - window_size + 1, window_size):
        window = matrix[start:start + window_size]

        window_means.append(np.mean(window))

    window_means = np.array(window_means)

    print(f"\n{label}")
    print("Windows:", len(window_means))
    print("First 10 window means:")
    print(np.round(window_means[:10], 2))
    print("Overall mean:", round(np.mean(window_means), 2))
    print("Window variation:", round(np.std(window_means), 2))    

    print("Minimum:", round(np.min(window_means), 2))
    print("10th percentile:", round(np.percentile(window_means, 10), 2))
    print("Median:", round(np.median(window_means), 2))
    print("90th percentile:", round(np.percentile(window_means, 90), 2))
    print("Maximum:", round(np.max(window_means), 2))

