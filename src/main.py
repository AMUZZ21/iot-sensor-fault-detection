"""
main.py
========
End-to-end pipeline:
  1. Load data (real UCI batches if present in data/raw/, else synthetic demo data)
  2. Clean + split by batch (train = early batches, test = later batches -> drift stress-test)
  3. Inject synthetic faults into TRAIN and TEST separately (labels for supervised eval)
  4. Feature engineering
  5. Scale
  6. Train Isolation Forest, One-Class SVM, Random Forest, XGBoost
  7. Evaluate all four with Precision/Recall/F1/ROC-AUC/PR-AUC/False Alarm Rate
  8. Break down recall by fault type (spike/drift/noise/stuck)
  9. Check sensor-fault localization accuracy
  10. Save metrics table + plots to results/

USAGE
-----
    cd iot_fault_detection
    python src/main.py                      # uses data/raw/batch*.dat if present,
                                              # else falls back to synthetic demo data
    python src/main.py --use-synthetic       # force synthetic demo data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all_batches, make_synthetic_demo_dataset, FEATURE_NAMES
from fault_injection import inject_faults
from preprocessing import clean, batch_train_test_split, scale_features
from feature_engineering import add_engineered_features, engineered_feature_names, N_SENSORS
from models import get_all_models
from evaluate import evaluate_all, per_fault_type_recall, sensor_localization_accuracy, \
    plot_roc_pr_curves, plot_metric_comparison

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run(use_synthetic: bool, fault_fraction: float = 0.15, seed: int = 42):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- 1. Load ----
    if use_synthetic or not os.path.isdir(RAW_DIR) or not any(
        f.startswith("batch") for f in os.listdir(RAW_DIR) if os.path.isdir(RAW_DIR)
    ):
        print("Using SYNTHETIC demo data (place real batch*.dat in data/raw/ to use real data).")
        df = make_synthetic_demo_dataset(n_samples=6000, n_batches=10, seed=seed)
    else:
        print("Loading real UCI Gas Sensor Array Drift Dataset from data/raw/ ...")
        df = load_all_batches(RAW_DIR)

    df = clean(df, FEATURE_NAMES)
    print(f"Loaded dataset: {df.shape}, batches present: {sorted(df['batch'].unique())}")

    # ---- 2. Split by batch (time-based, simulates real drift deployment) ----
    all_batches = sorted(df["batch"].unique())
    n_train_batches = max(1, int(len(all_batches) * 0.7))
    train_batches = all_batches[:n_train_batches]
    test_batches = all_batches[n_train_batches:]
    print(f"Train batches: {train_batches} | Test batches: {test_batches}")

    train_df, test_df = batch_train_test_split(df, train_batches, test_batches)

    # ---- 3. Inject synthetic faults (separately, different seeds) ----
    train_faulted = inject_faults(train_df, FEATURE_NAMES, fault_fraction=fault_fraction, seed=seed)
    test_faulted = inject_faults(test_df, FEATURE_NAMES, fault_fraction=fault_fraction, seed=seed + 1)

    # ---- 4. Feature engineering ----
    train_feat = add_engineered_features(train_faulted, FEATURE_NAMES)
    test_feat = add_engineered_features(test_faulted, FEATURE_NAMES)
    eng_cols = engineered_feature_names(FEATURE_NAMES)
    model_feature_cols = FEATURE_NAMES + eng_cols

    # ---- 5. Scale ----
    train_scaled, test_scaled, scaler = scale_features(train_feat, test_feat, model_feature_cols)

    X_train, y_train = train_scaled[model_feature_cols].values, train_scaled["label"].values
    X_test, y_test = test_scaled[model_feature_cols].values, test_scaled["label"].values

    # Unsupervised models are trained mostly on "normal" data for a cleaner
    # notion of normality (best practice for IsoForest / OCSVM):
    X_train_normal = train_scaled.loc[train_scaled["label"] == 0, model_feature_cols].values

    contamination_estimate = y_train.mean()
    print(f"Train fault rate: {contamination_estimate:.3f} | Test fault rate: {y_test.mean():.3f}")

    # ---- 6. Train models ----
    models = get_all_models(contamination_estimate=contamination_estimate)
    results = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        if name in ("IsolationForest", "OneClassSVM"):
            model.fit(X_train_normal)
        else:
            model.fit(X_train, y_train)
        score = model.score(X_test)
        pred = model.predict(X_test)
        results[name] = (score, pred)

    # ---- 7. Evaluate ----
    metrics_df = evaluate_all(results, y_test)
    print("\n=== Model comparison ===")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)

    # ---- 8. Per-fault-type recall ----
    fault_type_df = per_fault_type_recall(test_scaled["fault_type"], results)
    print("\n=== Recall by fault type ===")
    print(fault_type_df.to_string(index=False))
    fault_type_df.to_csv(os.path.join(RESULTS_DIR, "recall_by_fault_type.csv"), index=False)

    # ---- 9. Sensor localization (using the engineered disagreement signal) ----
    # Rebuild the per-sensor disagreement matrix for the test set to check
    # whether the argmax sensor matches the truly-faulted sensor.
    X_test_raw = test_feat[FEATURE_NAMES].values
    sensor_means = np.zeros((len(test_feat), N_SENSORS))
    for s in range(N_SENSORS):
        block = X_test_raw[:, s * 8:(s + 1) * 8]
        sensor_means[:, s] = block.mean(axis=1)
    row_median = np.median(sensor_means, axis=1, keepdims=True)
    row_mad = np.median(np.abs(sensor_means - row_median), axis=1, keepdims=True) + 1e-6
    disagreement_matrix = np.abs(sensor_means - row_median) / row_mad

    print("\n=== Sensor fault localization accuracy (top-1) ===")
    for name, (score, pred) in results.items():
        acc, n_detected = sensor_localization_accuracy(
            y_test, test_scaled["faulty_sensor"].values, disagreement_matrix, pred
        )
        print(f"{name}: top-1 sensor localization accuracy = {acc:.3f} "
              f"(over {n_detected} correctly-detected anomalies)")

    # ---- 10. Plots ----
    plot_roc_pr_curves(results, y_test, save_path=os.path.join(RESULTS_DIR, "roc_pr_curves.png"))
    plot_metric_comparison(metrics_df, save_path=os.path.join(RESULTS_DIR, "metric_comparison.png"))

    print(f"\nAll results saved to {RESULTS_DIR}/")
    return metrics_df, fault_type_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-synthetic", action="store_true",
                         help="Force use of synthetic demo data even if real data is present.")
    parser.add_argument("--fault-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.use_synthetic, args.fault_fraction, args.seed)
