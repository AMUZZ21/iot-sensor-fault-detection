"""
feature_engineering.py
=========================
Builds extra features on top of the 128 raw sensor descriptors that make
abnormal behavior (drift / spike / noise / stuck) easier for the models to
separate from normal readings.

Because the 128 raw features are 16 sensors x 8 descriptors each, most
engineered features operate PER SENSOR (across its 8 descriptors) and then
get aggregated ACROSS the 16 sensors -- this is what lets us later ask
"which sensor looks most anomalous" for the fault-localization stretch goal.
"""

import numpy as np
import pandas as pd

N_SENSORS = 16
N_DESC_PER_SENSOR = 8


def _sensor_block(X: np.ndarray, sensor_idx: int) -> np.ndarray:
    start = sensor_idx * N_DESC_PER_SENSOR
    return X[:, start:start + N_DESC_PER_SENSOR]


def add_engineered_features(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """
    Adds, per row:
      - per-sensor mean/std/max/min across that sensor's 8 descriptors
        (sensor_{i}_mean, sensor_{i}_std, ...) for i in 0..15
      - cross-sensor stats of those per-sensor means (how much sensors
        disagree with each other -- useful since faulty sensors deviate
        from the consensus of the other 15)
      - a simple z-score-based outlier count per row (how many of the 128
        features are >3 std from the column's mean, using this dataframe's
        own stats -- a cheap, explainable anomaly signal that complements
        the ML models)
    Returns a NEW dataframe with extra columns appended.
    """
    out = df.copy()
    X = df[feature_cols].values.astype(float)

    sensor_means = np.zeros((len(df), N_SENSORS))
    sensor_stds = np.zeros((len(df), N_SENSORS))
    for s in range(N_SENSORS):
        block = _sensor_block(X, s)
        sensor_means[:, s] = block.mean(axis=1)
        sensor_stds[:, s] = block.std(axis=1)
        out[f"sensor_{s}_mean"] = sensor_means[:, s]
        out[f"sensor_{s}_std"] = sensor_stds[:, s]

    # Cross-sensor agreement: does one sensor's mean look far from the
    # median of all 16 sensors' means, for THIS row?
    row_median = np.median(sensor_means, axis=1, keepdims=True)
    row_mad = np.median(np.abs(sensor_means - row_median), axis=1, keepdims=True) + 1e-6
    sensor_disagreement = np.abs(sensor_means - row_median) / row_mad  # (n, 16)
    out["max_sensor_disagreement"] = sensor_disagreement.max(axis=1)
    out["argmax_disagreement_sensor"] = sensor_disagreement.argmax(axis=1)

    # Simple global z-score outlier count (uses this df's own column stats;
    # for train/test, compute on train only and reuse -- see main.py)
    col_mean = X.mean(axis=0)
    col_std = X.std(axis=0) + 1e-6
    z = np.abs((X - col_mean) / col_std)
    out["n_features_gt_3std"] = (z > 3).sum(axis=1)
    out["max_abs_zscore"] = z.max(axis=1)

    return out


def engineered_feature_names(base_feature_cols):
    names = []
    for s in range(N_SENSORS):
        names += [f"sensor_{s}_mean", f"sensor_{s}_std"]
    names += ["max_sensor_disagreement", "argmax_disagreement_sensor",
              "n_features_gt_3std", "max_abs_zscore"]
    return names
