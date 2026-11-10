import csv
import ast
import time
from collections import deque

import serial
import numpy as np
import matplotlib.pyplot as plt

# ESP32 connection
PORT = "COM7"
BAUD_RATE = 921600

# Display settings
WINDOW_SIZE = 100
NUM_SUBCARRIERS = 64

data_buffer = deque(maxlen=WINDOW_SIZE)
received = 0
malformed = 0

fig, ax = plt.subplots(figsize=(12, 6))

heatmap = ax.imshow(
    np.zeros((NUM_SUBCARRIERS, WINDOW_SIZE)),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
    vmin=0,
    vmax=100
)

ax.set_title("Live Wi-Fi CSI Heatmap")
ax.set_xlabel("Recent CSI Samples")
ax.set_ylabel("Subcarrier")

colorbar = fig.colorbar(heatmap, ax=ax)
colorbar.set_label("CSI Magnitude")

status = ax.text(
    0.02, 0.98, "Waiting for CSI data...",
    transform=ax.transAxes,
    color="white",
    va="top"
)

plt.tight_layout()


def parse_csi_line(line):
    row = next(csv.reader([line]))

    if len(row) != 25 or row[0] != "CSI_DATA":
        return None

    values = ast.literal_eval(row[-1])
    iq = np.array(values, dtype=float).reshape(-1, 2)

    magnitude = np.sqrt(
        iq[:, 0] ** 2 + iq[:, 1] ** 2
    )

    if len(magnitude) != NUM_SUBCARRIERS:
        return None

    return magnitude


def update_plot():
    if data_buffer:
        matrix = np.array(data_buffer).T

        heatmap.set_data(matrix)
        heatmap.set_clim(
            vmin=max(0, np.percentile(matrix, 5)),
            vmax=max(1, np.percentile(matrix, 95))
        )

        status.set_text(
            f"CSI samples received: {received}\n"
            f"Invalid rows: {malformed}"
        )

    fig.canvas.draw_idle()


def main():
    global received, malformed

    print(f"Connecting to {PORT} at {BAUD_RATE} baud...")

    try:
        with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2)
            ser.reset_input_buffer()

            print("Connected. Live heatmap is running.")
            print("Close the heatmap window to stop.")

            plt.show(block=False)

            while plt.fignum_exists(fig.number):
                raw = ser.readline()

                if not raw:
                    plt.pause(0.01)
                    continue

                line = raw.decode(
                    "utf-8", errors="replace"
                ).strip()

                if not line.startswith("CSI_DATA,"):
                    continue

                try:
                    magnitude = parse_csi_line(line)
                    if magnitude is None:
                        malformed += 1
                        continue

                    data_buffer.append(magnitude)
                    received += 1

                    if received % 5 == 0:
                        update_plot()

                except Exception:
                    malformed += 1

                plt.pause(0.001)

    except serial.SerialException as exc:
        print("Serial connection error:", exc)
        print("Check COM7 and close any other program using it.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        plt.close("all")
        print("Live heatmap stopped.")


if __name__ == "__main__":
    main()