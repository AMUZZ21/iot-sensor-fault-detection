"""
data_loader.py
================
Loads the UCI Gas Sensor Array Drift Dataset.

DATASET FORMAT NOTE
--------------------
The UCI "Gas Sensor Array Drift Dataset" (https://archive.ics.uci.edu/dataset/224)
ships as 10 files: batch1.dat ... batch10.dat, in LIBSVM sparse-format text, e.g.:

    1;1 1:15596.16 2:1.868245 3:2.201091 ... 128:11.15132

- The token before ';' is the GAS CLASS LABEL (1-6: the gas that was presented,
  NOT a fault label). Each subsequent "index:value" pair is one of 128 features
  (16 sensors x 8 steady-state/transient descriptors per sensor).
- Batches roughly correspond to time (batch1 = earliest months, batch10 = latest),
  which is what produces the "drift" -- the sensors' response distribution
  shifts over the 36-month collection period.

IMPORTANT FOR THIS PROJECT: this dataset has NO ground-truth "sensor fault" labels.
The gas-class label tells you *which gas* was sniffed, not whether a sensor was
faulty. So we use it in two ways (see feature_engineering.py / main.py):
  1. UNSUPERVISED framing: train anomaly detectors (Isolation Forest, One-Class
     SVM) on "normal" batches, treat rare/extreme points as anomalies, and use
     the batch-to-batch drift itself as a natural stress-test for the models.
  2. SEMI-SYNTHETIC SUPERVISED framing: to get precision/recall/F1/ROC-AUC
     numbers (which need real labels), we programmatically INJECT synthetic
     faults (spikes, drift, noise, stuck-at) into a held-out split and label
     those points as anomalies (1) vs. normal (0). This is standard practice
     in sensor-fault-detection literature when no natively-labeled fault
     dataset is available, and it lets Random Forest / XGBoost be trained
     and evaluated in a fully supervised way.

Download the data (manually, since this sandbox can't reach archive.ics.uci.edu):
    1. Go to https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
    2. Download and unzip "Dataset.zip" -> you get batch1.dat ... batch10.dat
    3. Place them in <project_root>/data/raw/
"""

import os
import re
import numpy as np
import pandas as pd
from glob import glob

N_FEATURES = 128
FEATURE_NAMES = [f"f{i}" for i in range(1, N_FEATURES + 1)]


def _parse_libsvm_line(line: str, line_no: int = -1, filepath: str = ""):
    """
    Parse UCI Gas Sensor Array Drift dataset lines.

    Supports both:
        label;concentration 1:value 2:value ...
    and:
        label 1:value 2:value 3:value ...

    Returns:
        (label, concentration, features)
    or None for malformed/blank lines.
    """
    raw = line
    line = line.strip()

    if not line:
        return None

    try:
        # ---------------------------------------------------------
        # FORMAT 1:
        # label;concentration 1:value 2:value ...
        # ---------------------------------------------------------
        if ";" in line:
            label_part, rest = line.split(";", 1)
            label = int(float(label_part.strip()))

            feats = np.zeros(N_FEATURES, dtype=np.float64)
            concentration = np.nan

            for tok in rest.strip().split():
                if ":" in tok:
                    idx_str, val_str = tok.split(":", 1)
                    idx = int(idx_str) - 1

                    if 0 <= idx < N_FEATURES:
                        feats[idx] = float(val_str)
                else:
                    # Non-feature token = concentration
                    concentration = float(tok)

            return label, concentration, feats

        # ---------------------------------------------------------
        # FORMAT 2:
        # label 1:value 2:value 3:value ...
        # ---------------------------------------------------------
        else:
            tokens = line.split()

            if len(tokens) < 2:
                print(
                    f"WARNING: skipping malformed line in {filepath} "
                    f"at line {line_no}: {raw!r}"
                )
                return None

            label = int(float(tokens[0]))

            feats = np.zeros(N_FEATURES, dtype=np.float64)
            concentration = np.nan

            for tok in tokens[1:]:
                if ":" not in tok:
                    continue

                idx_str, val_str = tok.split(":", 1)
                idx = int(idx_str) - 1

                if 0 <= idx < N_FEATURES:
                    feats[idx] = float(val_str)

            return label, concentration, feats

    except (ValueError, IndexError) as e:
        print(
            f"WARNING: skipping malformed line in {filepath} "
            f"at line {line_no}: {raw!r} ({e})"
        )
        return None


def load_batch(filepath: str, batch_id: int) -> pd.DataFrame:
    """Load a single batchN.dat file into a DataFrame."""

    rows = []
    n_skipped = 0

    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):

            parsed = _parse_libsvm_line(
                line,
                line_no,
                filepath
            )

            if parsed is None:
                n_skipped += 1
                continue

            label, concentration, feats = parsed

            rows.append(
                [batch_id, label, concentration]
                + feats.tolist()
            )

    if n_skipped:
        print(
            f"  -> skipped {n_skipped} malformed/blank "
            f"line(s) in {filepath}"
        )

    cols = [
        "batch",
        "gas_class",
        "concentration"
    ] + FEATURE_NAMES

    return pd.DataFrame(rows, columns=cols)


def load_all_batches(raw_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all batchN.dat files found in raw_dir.
    Returns a single DataFrame sorted by batch (i.e. roughly chronological).
    """
    paths = sorted(
        glob(os.path.join(raw_dir, "batch*.dat")),
        key=lambda p: int(re.search(r"batch(\d+)", os.path.basename(p)).group(1)),
    )
    if not paths:
        raise FileNotFoundError(
            f"No batch*.dat files found in {raw_dir}. "
            "Download the dataset from UCI and place batch1.dat..batch10.dat there "
            "(see the module docstring in data_loader.py)."
        )
    frames = []
    for p in paths:
        batch_id = int(re.search(r"batch(\d+)", os.path.basename(p)).group(1))
        frames.append(load_batch(p, batch_id))
        print(f"Loaded {p}: {frames[-1].shape}")
    df = pd.concat(frames, ignore_index=True)
    return df


def make_synthetic_demo_dataset(n_samples: int = 4000, n_batches: int = 10,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset that MIMICS the structure and drift behavior
    of the UCI Gas Sensor Array Drift Dataset (128 features, 16 "sensors" x 8
    descriptors, 6 gas classes, 10 batches with gradual drift).

    Use this ONLY to test/demo the pipeline end-to-end before the real data is
    downloaded. Swap it out for load_all_batches() once batch*.dat files are
    in data/raw/. The whole rest of the pipeline (preprocessing, feature
    engineering, models, evaluation) works identically on either.
    """
    rng = np.random.default_rng(seed)
    n_sensors = 16
    n_desc = 8  # descriptors per sensor (steady-state + transient features)
    rows = []
    for i in range(n_samples):
        batch = rng.integers(1, n_batches + 1)
        gas_class = rng.integers(1, 7)
        drift = (batch - 1) * 0.08  # slow multiplicative drift over batches
        base = rng.normal(loc=5 + gas_class, scale=1.0, size=n_sensors)
        sensor_block = []
        for s in range(n_sensors):
            resp = base[s] * (1 + drift) + rng.normal(0, 0.3)
            descriptors = resp * rng.uniform(0.7, 1.3, size=n_desc)
            sensor_block.append(descriptors)
        feats = np.concatenate(sensor_block)
        rows.append([batch, gas_class] + feats.tolist())
    cols = ["batch", "gas_class"] + FEATURE_NAMES
    df = pd.DataFrame(rows, columns=cols)
    return df


if __name__ == "__main__":
    demo = make_synthetic_demo_dataset(500)
    print(demo.head())
    print(demo.shape)
