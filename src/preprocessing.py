"""
preprocessing.py
==================
Cleaning and scaling utilities.

Key design choice: we split TRAIN/TEST BY BATCH (time), not randomly.
This matters a lot for a drift dataset -- a random split would let the model
"see the future" (later-batch statistics leaking into training) and give
unrealistically optimistic results. Splitting by batch mimics the real
deployment scenario: train on early sensor behavior, detect faults/drift on
later, unseen time periods.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Drop duplicate rows and rows with NaN/inf in feature columns."""
    df = df.drop_duplicates().reset_index(drop=True)
    mask = np.isfinite(df[feature_cols].values).all(axis=1)
    dropped = (~mask).sum()
    if dropped:
        print(f"Dropping {dropped} rows with NaN/inf values")
    return df.loc[mask].reset_index(drop=True)


def batch_train_test_split(df: pd.DataFrame, train_batches, test_batches):
    """Split by the 'batch' column into train/test (no shuffling across time)."""
    train_df = df[df["batch"].isin(train_batches)].reset_index(drop=True)
    test_df = df[df["batch"].isin(test_batches)].reset_index(drop=True)
    return train_df, test_df


def scale_features(train_df, test_df, feature_cols):
    """
    Fit StandardScaler on TRAIN ONLY, apply to both. Returns scaled copies
    plus the fitted scaler (needed later at inference time on new sensor
    readings).
    """
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, scaler
