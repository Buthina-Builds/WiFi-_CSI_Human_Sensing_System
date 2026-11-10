import ast
import csv
import time
import serial
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from matplotlib.patches import FancyBboxPatch

PORT = "COM7"
BAUD_RATE = 921600

NUM_SUBCARRIERS = 64
HEATMAP_SIZE = 100
GRAPH_SIZE = 200
UPDATE_EVERY = 5

heatmap_buffer = deque(maxlen=HEATMAP_SIZE)
activity_buffer = deque(maxlen=GRAPH_SIZE)

# =========================
# ML CONFIGURATION
# =========================

import joblib

MODEL_ARTIFACT = joblib.load("college_csi_model.joblib")
ml_model = MODEL_ARTIFACT["model"]
ML_FEATURES = MODEL_ARTIFACT["features"]

WINDOW_SIZE = 97

VALID_SUBCARRIERS = [
    i for i in range(64)
    if i not in [0, 1] and not 28 <= i <= 36
]

ml_window = deque(maxlen=WINDOW_SIZE)

current_prediction = "WAITING"

# =========================
# ML FEATURE EXTRACTION
# =========================

def extract_live_features(window):

    csi_matrix = np.array([item[0] for item in window])
    rssi_values = np.array([item[1] for item in window])

    features = {}

    # Subcarrier statistics
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
    features["temporal_change"] = np.mean(
        np.abs(np.diff(row_means))
    )

    features["window_variation"] = np.std(row_means)

    # RSSI statistics
    features["rssi_mean"] = np.mean(rssi_values)
    features["rssi_std"] = np.std(rssi_values)

    # Match training feature order
    feature_vector = np.array([
        features[name] for name in ML_FEATURES
    ]).reshape(1, -1)

    return feature_vector

# =========================
# ML PREDICTION
# =========================

def predict_window(window):

    if len(window) < WINDOW_SIZE:
        return "WAITING"

    feature_vector = extract_live_features(window)

    prediction = ml_model.predict(feature_vector)[0]

    return prediction


previous_magnitude = None
received = 0
malformed = 0

# =========================
# DASHBOARD DESIGN
# =========================

plt.ion()

fig = plt.figure(figsize=(14, 9), facecolor="#111827")

grid = fig.add_gridspec(
    2, 2,
    height_ratios=[2.2, 1],
    width_ratios=[3.2, 1.3],
    hspace=0.30,
    wspace=0.18
)

ax_heat = fig.add_subplot(grid[0, 0])
ax_status = fig.add_subplot(grid[0, 1])
ax_graph = fig.add_subplot(grid[1, :])

fig.suptitle(
    "WI-FI CSI HUMAN SENSING SYSTEM",
    fontsize=19,
    fontweight="bold",
    color="white",
    y=0.97
)

connection_text = fig.text(
    0.5, 0.925,
    "● CONNECTING TO ESP32...",
    ha="center",
    fontsize=10,
    color="#FBBF24"
)

# =========================
# HEATMAP
# =========================

ax_heat.set_facecolor("#1F2937")

heatmap = ax_heat.imshow(
    np.zeros((NUM_SUBCARRIERS, HEATMAP_SIZE)),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest"
)

ax_heat.set_title(
    "LIVE CSI HEATMAP",
    color="white",
    fontweight="bold",
    pad=12
)

ax_heat.set_xlabel("Recent Samples", color="white")
ax_heat.set_ylabel("Subcarrier", color="white")

ax_heat.tick_params(colors="white")

for spine in ax_heat.spines.values():
    spine.set_color("#374151")

colorbar = fig.colorbar(heatmap, ax=ax_heat)
colorbar.set_label("CSI Magnitude", color="white")
colorbar.ax.tick_params(colors="white")

# =========================
# DETECTION STATUS CARDS
# =========================

ax_status.set_facecolor("#1F2937")
ax_status.axis("off")

ax_status.set_title(
    "DETECTION CLASSES",
    color="white",
    fontweight="bold",
    pad=12
)

class_cards = [
    ("NO PERSON", "#6B7280"),
    ("STATIONARY", "#D97706"),
    ("MOVING", "#059669")
]

card_status_text = {}

for i, (name, color) in enumerate(class_cards):

    y = 0.72 - i * 0.29

    card = FancyBboxPatch(
        (0.07, y),
        0.86,
        0.22,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        transform=ax_status.transAxes,
        facecolor="#273449",
        edgecolor=color,
        linewidth=2
    )

    ax_status.add_patch(card)

    ax_status.text(
        0.5, y + 0.145,
        name,
        transform=ax_status.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white"
    )
    status_text = ax_status.text(
    0.5, y + 0.065,
    "WAITING",
    transform=ax_status.transAxes,
    ha="center",
    va="center",
    fontsize=8,
    color="#D1D5DB"
)

card_status_text[name] = status_text

# =========================
# ACTIVITY GRAPH
# =========================

ax_graph.set_facecolor("#1F2937")

line, = ax_graph.plot(
    [],
    [],
    color="#38BDF8",
    linewidth=2
)

ax_graph.set_title(
    "CSI ACTIVITY LEVEL",
    color="white",
    fontweight="bold",
    pad=10
)

ax_graph.set_xlabel("Recent Samples", color="white")
ax_graph.set_ylabel("Mean CSI Change", color="white")

ax_graph.tick_params(colors="white")
ax_graph.grid(True, alpha=0.2)

for spine in ax_graph.spines.values():
    spine.set_color("#374151")

# =========================
# COUNTERS
# =========================

counter_text = fig.text(
    0.03, 0.025,
    "CSI Samples: 0 | Invalid Rows: 0",
    fontsize=10,
    color="#D1D5DB"
)

classifier_text = fig.text(
    0.97, 0.025,
    "CLASSIFIER: PENDING",
    ha="right",
    fontsize=10,
    color="#FBBF24"
)

plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.90])


# =========================
# CSI PARSING
# =========================
def parse_csi(line_text):

    row = next(csv.reader([line_text]))

    if len(row) != 25 or row[0] != "CSI_DATA":
        return None

    rssi = float(row[3])

    values = ast.literal_eval(row[-1])

    iq = np.array(values, dtype=float).reshape(-1, 2)

    magnitude = np.sqrt(
        iq[:, 0] ** 2 + iq[:, 1] ** 2
    )

    if len(magnitude) != NUM_SUBCARRIERS:
        return None

    return magnitude, rssi

# =========================
# UPDATE DASHBOARD
# =========================

def update_dashboard():

    if not heatmap_buffer:
        return

    samples = list(heatmap_buffer)

    matrix = np.zeros(
        (NUM_SUBCARRIERS, HEATMAP_SIZE)
    )

    matrix[:, -len(samples):] = np.array(samples).T

    heatmap.set_data(matrix)

    heatmap.set_clim(
        vmin=max(0, np.percentile(matrix, 5)),
        vmax=max(1, np.percentile(matrix, 95))
    )

    activity = list(activity_buffer)

    x = np.arange(len(activity))

    line.set_data(x, activity)

    ax_graph.set_xlim(
        0, max(GRAPH_SIZE - 1, len(activity) - 1)
    )

    if activity:
        upper = max(activity) * 1.2
        ax_graph.set_ylim(0, max(upper, 1))

    counter_text.set_text(
        f"CSI Samples: {received} | Invalid Rows: {malformed}"
    )
    # Update ML detection cards
    prediction_names = {
        "NO_PERSON": "NO PERSON",
        "STATIONARY": "STATIONARY",
        "MOVING": "MOVING"
    }

    for name, status_text in card_status_text.items():

        if current_prediction == "WAITING":
            status_text.set_text("WAITING")
            status_text.set_color("#D1D5DB")

        elif prediction_names.get(current_prediction) == name:
            status_text.set_text("DETECTED")
            status_text.set_color("#34D399")

        else:
            status_text.set_text("INACTIVE")
            status_text.set_color("#9CA3AF")

    classifier_text.set_text(
        f"CLASSIFIER: {current_prediction}"
    )
    fig.canvas.draw_idle()


# =========================
# MAIN
# =========================

def main():

    global previous_magnitude
    global received, malformed
    global current_prediction

    print(f"Connecting to {PORT} at {BAUD_RATE} baud...")

    try:

        with serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=1
        ) as ser:

            time.sleep(2)
            ser.reset_input_buffer()

            connection_text.set_text(
                "● ESP32 CONNECTED"
            )
            connection_text.set_color("#34D399")

            print("ESP32 connected successfully.")
            print("Dashboard is running.")
            print("Close the window to stop.")

            while plt.fignum_exists(fig.number):

                raw = ser.readline()

                if not raw:
                    plt.pause(0.01)
                    continue

                text_line = raw.decode(
                    "utf-8",
                    errors="replace"
                ).strip()

                if not text_line.startswith("CSI_DATA,"):
                    continue

                try:

                    parsed = parse_csi(text_line)

                    if parsed is None:
                        malformed += 1
                        continue

                    magnitude, rssi = parsed

                    heatmap_buffer.append(magnitude)

                    # Collect CSI readings for ML
                    ml_window.append((magnitude, rssi))

                    if len(ml_window) == WINDOW_SIZE:

                        current_prediction = predict_window(
                            list(ml_window)
                        )

                        print("ML PREDICTION:", current_prediction)

                        ml_window.clear()

                    if previous_magnitude is not None:

                        change = np.mean(
                            np.abs(
                                magnitude - previous_magnitude
                            )
                        )

                        activity_buffer.append(change)

                    previous_magnitude = magnitude
                    received += 1                
                

                    if received % UPDATE_EVERY == 0:
                        update_dashboard()

                except Exception:
                    malformed += 1

                plt.pause(0.001)

    except serial.SerialException as exc:

        connection_text.set_text(
            "● ESP32 CONNECTION ERROR"
        )
        connection_text.set_color("#EF4444")

        print("Serial connection error:", exc)
        print("Check COM7 and close other programs using it.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        plt.close("all")
        print("Dashboard stopped.")


if __name__ == "__main__":
    main()


    