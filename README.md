Wi-Fi CSI Human Sensing System

Author: Bothina Hussein

Overview

The Wi-Fi CSI Human Sensing System is a machine-learning-based prototype that uses Channel State Information (CSI) from Wi-Fi signals to detect human presence and recognize activity states without using a camera.

The system analyzes changes in Wi-Fi signal characteristics to classify the surrounding environment into three states:

- NO_PERSON: No person detected.
- STATIONARY: A person is present but not moving.
- MOVING: A person is moving.

How It Works

The system follows this processing pipeline:

1. An ESP32-C6 captures Wi-Fi CSI data.
2. CSI records are transmitted to a computer through a USB serial connection.
3. Python processes the incoming CSI data.
4. Signal magnitude and statistical features are extracted.
5. A trained machine learning model predicts the current activity state.
6. A live dashboard displays the CSI visualization and prediction.

System Architecture

Wi-Fi Router
     |
     v
ESP32-C6 (CSI Acquisition)
     |
     v
USB Serial Communication
     |
     v
Python Data Processing
     |
     v
Feature Extraction
     |
     v
Extra Trees Classifier
     |
     v
Live Visualization Dashboard

Technologies Used

- Python
- NumPy
- Scikit-learn
- Matplotlib
- PySerial
- Joblib
- ESP32-C6
- Wi-Fi Channel State Information (CSI)

Machine Learning Model

The project uses an Extra Trees Classifier for three-class classification.

Model configuration:

- Number of estimators: 500
- Number of input features: 114
- Window size: 97 CSI records
- Class weighting: Balanced
- Minimum samples per leaf: 2

The trained model is saved as:

"college_csi_model.joblib"

The model artifact contains the trained classifier, feature names, and class labels.

Hardware Requirements

- ESP32-C6 development board
- Wi-Fi router
- USB cable
- Computer running Python

Software Requirements

- Python 3.12
- Required Python libraries listed in "requirements.txt"

Installation

Clone the repository:

git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Configuration

Before running the dashboard:

1. Connect the ESP32-C6 to your computer.
2. Identify the correct serial port.
3. Open "live_csi_dashboard.py".
4. Update the serial port setting:

PORT = "COM7"

Replace "COM7" with the port assigned to your ESP32-C6.

The baud rate is configured as:

BAUD_RATE = 921600

Make sure "college_csi_model.joblib" is in the same directory as the dashboard script.

Running the Project

Run the live dashboard:

python live_csi_dashboard.py

The dashboard receives CSI data, visualizes signal changes, and displays the predicted human activity state.

Project Features

- Real-time CSI data acquisition.
- Wi-Fi signal magnitude visualization.
- Live signal monitoring.
- Machine learning-based human activity classification.
- Three prediction states: NO_PERSON, STATIONARY, and MOVING.
- Serial communication with ESP32-C6.

Limitations

- This is a prototype and currently requires a computer for Python processing, machine learning inference, and dashboard visualization.
- Prediction response time depends on CSI data acquisition and processing.
- The system may be affected by changes in the surrounding environment.
- The CSI heatmap represents signal magnitude, not a real thermal image or human body image.
- Further testing and optimization are needed to improve response speed and reliability.

Dataset

Raw CSI recordings are not included in this repository.

The dataset is excluded to keep the repository lightweight. Users can collect their own CSI recordings using the data collection scripts.

Future Improvements

- Reduce prediction latency.
- Improve classification reliability.
- Test across more environments.
- Optimize real-time data processing.
- Explore standalone deployment options.

---

Developed by Bothina Hussein