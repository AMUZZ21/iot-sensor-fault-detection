"""
fault_injection.py
====================
Injects synthetic sensor faults into a clean dataset and produces ground-truth
labels, so that supervised models (Random Forest, XGBoost) and standard metrics
(Precision/Recall/F1/ROC-AUC) can be computed. This mirrors how real IoT fault
literature evaluates unsupervised detectors when no natively fault-labeled
dataset exists: inject known fault types into real sensor traces, then check
whether the detector recovers them.

Fault types implemented (each acts on a random subset of sensor-feature
columns, for a random subset of rows, to simulate one physical sensor going
bad while its neighbours stay normal):

  - spike   : a short, large-magnitude deviation ("sudden spike")
  - drift   : a slow additive/multiplicative bias that grows in one direction
  - noise   : injected high-variance Gaussian noise on top of the real signal
  - stuck   : the sensor "freezes" and repeats a constant value regardless
              of the true underlying signal

Each faulted ROW is labeled y=1 (anomalous); untouched rows are y=0 (normal).
We also keep a `fault_type` column and a `faulty_sensor` column (which of the
16 sensors was corrupted) so we can later evaluate "can the model localize
WHICH sensor is at fault" (the project's stretch goal).
"""

import numpy as np
import pandas as pd

N_SENSORS = 16
N_DESC_PER_SENSOR = 8


def _sensor_col_indices(sensor_idx: int):
    """Feature-column indices (0-based, within f1..f128) belonging to one sensor."""
    start = sensor_idx * N_DESC_PER_SENSOR
    return list(range(start, start + N_DESC_PER_SENSOR))


def inject_faults(df: pd.DataFrame, feature_cols, fault_fraction: float = 0.15,
                   seed: int = 42) -> pd.DataFrame:
    """
    Returns a NEW dataframe (copy of df) with:
      - a fraction `fault_fraction` of rows corrupted by a randomly chosen
        fault type on a randomly chosen sensor's descriptor columns
      - column 'label': 1 = faulty/anomalous, 0 = normal
      - column 'fault_type': one of {'none','spike','drift','noise','stuck'}
      - column 'faulty_sensor': sensor index (0-15) that was corrupted, or -1
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)
    n_faulty = int(n * fault_fraction)
    fault_idx = rng.choice(n, size=n_faulty, replace=False)

    out["label"] = 0
    out["fault_type"] = "none"
    out["faulty_sensor"] = -1

    fault_types = ["spike", "drift", "noise", "stuck"]
    X = out[feature_cols].values.astype(float)
    col_std = X.std(axis=0) + 1e-6  # per-feature scale, for realistic magnitudes

    for row_i in fault_idx:
        ftype = rng.choice(fault_types)
        sensor_idx = rng.integers(0, N_SENSORS)
        cols = _sensor_col_indices(sensor_idx)

        if ftype == "spike":
            magnitude = rng.uniform(4, 8)  # in units of feature std
            sign = rng.choice([-1, 1])
            X[row_i, cols] += sign * magnitude * col_std[cols]

        elif ftype == "drift":
            # simulate a bias that has been accumulating -- represented here
            # as a large additive offset proportional to feature scale
            bias = rng.uniform(2, 5) * col_std[cols]
            X[row_i, cols] += bias

        elif ftype == "noise":
            extra_noise = rng.normal(0, 3.0, size=len(cols)) * col_std[cols]
            X[row_i, cols] += extra_noise

        elif ftype == "stuck":
            # freeze all descriptors of this sensor to a fixed (wrong-ish) value
            stuck_value = rng.uniform(0.5, 1.5) * np.nanmean(X[:, cols])
            X[row_i, cols] = stuck_value

        out.loc[row_i, "fault_type"] = ftype
        out.loc[row_i, "faulty_sensor"] = sensor_idx
        out.loc[row_i, "label"] = 1

    out[feature_cols] = X
    return out


if __name__ == "__main__":
    from data_loader import make_synthetic_demo_dataset, FEATURE_NAMES
    df = make_synthetic_demo_dataset(1000)
    faulted = inject_faults(df, FEATURE_NAMES, fault_fraction=0.15)
    print(faulted["label"].value_counts())
    print(faulted["fault_type"].value_counts())
