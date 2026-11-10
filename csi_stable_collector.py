"""
CSI Stable - Serial Data Collector
---------------------------------
Reads CSI_DATA rows from the ESP32-C6 csi_recv_router example
and stores them in a labeled CSV file.

Important:
- Connect the ESP32-C6 before running this script.
- Close ESP-IDF monitor before opening the same COM port here.
- This script COLLECTS data; it does not yet classify people.
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports

BAUD_RATE = 921600
EXPECTED_FIELDS = 25

CSI_COLUMNS = [
    "type", "seq", "mac", "rssi", "rate", "noise_floor",
    "field_6", "field_7", "field_8", "field_9",
    "field_10", "field_11", "field_12", "field_13",
    "field_14", "field_15", "field_16", "field_17",
    "field_18", "field_19", "field_20", "field_21",
    "field_22","field_23","data"
]

LABELS = {
    "1": "NO_PERSON",
    "2": "PERSON_STATIONARY",
    "3": "PERSON_MOVING",
}


def choose_port():
    ports = list(list_ports.comports())
    if not ports:
        print("\nNo serial ports found. Connect the ESP32-C6 and try again.")
        return None

    print("\nAvailable serial ports:")
    for i, port in enumerate(ports, start=1):
        print(f"  {i}. {port.device} - {port.description}")

    while True:
        choice = input("\nEnter COM port (example: COM5), or list number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(ports):
            return ports[int(choice) - 1].device
        if choice:
            return choice.upper()


def choose_label():
    print("\nChoose the condition you are recording:")
    print("  1. NO PERSON")
    print("  2. PERSON STATIONARY")
    print("  3. PERSON MOVING")
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in LABELS:
            return LABELS[choice]
        print("Please enter 1, 2, or 3.")


def main():
    print("=" * 58)
    print("             CSI STABLE - DATA COLLECTOR")
    print("=" * 58)

    port = choose_port()
    if port is None:
        return

    label = choose_label()
    duration_text = input("Recording duration in seconds (default 60): ").strip()
    duration = int(duration_text) if duration_text else 60
    if duration <= 0:
        print("Duration must be greater than zero.")
        return

    output_dir = Path("csi_dataset")
    output_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{label}_{stamp}.csv"

    print(f"\nPort: {port}")
    print(f"Baud rate: {BAUD_RATE}")
    print(f"Label: {label}")
    print(f"Duration: {duration} seconds")
    print(f"Output: {output_file.resolve()}")
    input("\nWhen ready, press ENTER to start recording...")

    count = 0
    malformed = 0
    start = time.time()

    try:
        with serial.Serial(port, BAUD_RATE, timeout=1) as ser, \
             output_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["host_timestamp", "label"] + CSI_COLUMNS)

            # Allow the serial connection/device to settle.
            time.sleep(2)
            ser.reset_input_buffer()

            print("\nRecording... Keep the room condition consistent.")
            print("Press Ctrl+C to stop early.\n")

            while time.time() - start < duration:
                raw = ser.readline()
                if not raw:
                    continue

                host_time = datetime.now().isoformat(timespec="milliseconds")
                line = raw.decode("utf-8", errors="replace").strip()

                if not line.startswith("CSI_DATA,"):
                    continue

                try:
                    row = next(csv.reader([line]))
                except (csv.Error, StopIteration):
                    malformed += 1
                    continue
                if len(row) != EXPECTED_FIELDS:
                 if malformed == 0:
                    print(f"\nDEBUG: Received fields = {len(row)}")
                    print(f"DEBUG: Expected fields = {EXPECTED_FIELDS}")
                    print("DEBUG row:", repr(row))
                    malformed += 1
                    continue

                if row[0] != "CSI_DATA":
                    malformed += 1
                    continue
                writer.writerow([host_time, label] + row)
                count += 1

                if count % 50 == 0:
                    file.flush()
                    elapsed = int(time.time() - start)
                    print(f"\rCaptured: {count} rows | elapsed: {elapsed}s", end="")

            file.flush()

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except serial.SerialException as exc:
        print(f"\nSerial error: {exc}")
        print("Check the COM port, USB cable, and whether another program is using the port.")
        return

    elapsed = time.time() - start
    print("\n\nRecording finished.")
    print(f"Valid CSI rows: {count}")
    print(f"Malformed CSI rows skipped: {malformed}")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Saved file: {output_file.resolve()}")

    if count == 0:
        print("\nNo CSI_DATA rows were received.")
        print("The ESP32 may not be connected to the hotspot, or the AP may not be sending matching frames.")


if __name__ == "__main__":
    main()
