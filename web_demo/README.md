# IoT Sensor Fault Detection — Working Professor Demo

This folder is meant to sit **inside your existing `iot_fault_detection` project root**.

## 1. Copy this folder
Put `web_demo` beside `src`, `data`, `models`, and `results`.

## 2. Install
From the project root:

    pip install -r web_demo/requirements.txt

## 3. Train + save the model
From the project root:

    python web_demo/train_and_save.py --fault-fraction 0.15

This creates:

    models/RandomForest_bundle.joblib

It saves the Random Forest, StandardScaler, feature names, and training statistics needed for inference.

For a faster test, use:

    python web_demo/train_and_save.py --fault-fraction 0.15 --estimators 100

## 4. Start the website

    python web_demo/app.py

Open http://127.0.0.1:5000

## 5. Demo to professor

1. Click **Load Normal Sample**.
2. Click **Run Prediction** → should show NORMAL.
3. Click **Load Faulty Sample**.
4. Click **Run Prediction** → should show FAULT / ANOMALY.
5. Explain: 128 raw sensor values → 164 engineered features → StandardScaler → Random Forest → classification.

Important: the UCI Gas Sensor Array Drift dataset itself does not contain sensor-fault ground truth. Your project injects synthetic spike/drift/noise/stuck faults and uses those injected labels for supervised evaluation.
