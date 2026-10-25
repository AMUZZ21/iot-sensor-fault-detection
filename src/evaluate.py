"""
evaluate.py
============
Computes the metrics named in the proposal (Precision, Recall, F1, ROC-AUC,
PR-AUC, False Alarm Rate) and produces comparison plots + a fault-type
breakdown + a sensor-localization accuracy check.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)


def false_alarm_rate(y_true, y_pred):
    """FAR = FP / (FP + TN): fraction of truly-normal points wrongly flagged."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def evaluate_model(name, y_true, y_score, y_pred):
    return {
        "model": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
    }


def evaluate_all(results: dict, y_true) -> pd.DataFrame:
    """
    results: {model_name: (y_score, y_pred)}
    Returns a metrics comparison DataFrame, sorted by F1 descending.
    """
    rows = [evaluate_model(name, y_true, score, pred)
            for name, (score, pred) in results.items()]
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return df


def per_fault_type_recall(y_true_fault_type, results: dict) -> pd.DataFrame:
    """
    Breaks down recall PER FAULT TYPE (spike/drift/noise/stuck) for each model.
    y_true_fault_type: pd.Series aligned with y_true, values in
        {'none','spike','drift','noise','stuck'}.
    results: {model_name: (y_score, y_pred)}
    """
    fault_types = [ft for ft in y_true_fault_type.unique() if ft != "none"]
    rows = []
    for model_name, (score, pred) in results.items():
        row = {"model": model_name}
        for ft in fault_types:
            mask = y_true_fault_type == ft
            if mask.sum() == 0:
                row[ft] = np.nan
                continue
            row[ft] = recall_score(
                np.ones(mask.sum()), np.asarray(pred)[mask.values], zero_division=0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def sensor_localization_accuracy(y_true, faulty_sensor_true, sensor_disagreement_matrix,
                                  predicted_anomaly_mask):
    """
    For rows correctly flagged as anomalous, checks whether the sensor with
    the highest 'disagreement' score (from feature_engineering.py) matches the
    sensor we actually injected the fault into. Returns top-1 localization
    accuracy over the correctly-detected anomalies.

    sensor_disagreement_matrix: array (n_samples, n_sensors) of per-sensor
        anomaly/disagreement scores (e.g. the sensor_disagreement matrix
        computed the same way as in feature_engineering.add_engineered_features,
        or per-sensor SHAP-style importance from a trained model).
    """
    detected = np.asarray(predicted_anomaly_mask).astype(bool) & (np.asarray(y_true) == 1)
    if detected.sum() == 0:
        return np.nan, 0
    predicted_sensor = sensor_disagreement_matrix[detected].argmax(axis=1)
    true_sensor = np.asarray(faulty_sensor_true)[detected]
    acc = (predicted_sensor == true_sensor).mean()
    return acc, int(detected.sum())


def plot_roc_pr_curves(results: dict, y_true, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, (score, _pred) in results.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend()

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    return fig


def plot_metric_comparison(metrics_df: pd.DataFrame, save_path=None):
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics_df))
    width = 0.15
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, metrics_df[m], width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(metrics_df["model"])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across Metrics")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    return fig
