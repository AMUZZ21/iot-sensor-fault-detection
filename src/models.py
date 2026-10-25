"""
models.py
==========
Wraps the four algorithms named in the proposal behind a common interface:
    fit(X_train, y_train=None)
    score(X)        -> higher = more anomalous (for ranking / PR-AUC / ROC-AUC)
    predict(X)       -> 0/1 hard label

Isolation Forest and One-Class SVM are UNSUPERVISED (fit on features only,
ideally on mostly-normal data). Random Forest and XGBoost are SUPERVISED
(need the injected fault labels from fault_injection.py).
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class IsoForestDetector:
    def __init__(self, contamination=0.1, n_estimators=200, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train=None):
        self.model.fit(X_train)
        return self

    def score(self, X):
        # sklearn's score_samples: higher = more normal, so flip sign
        return -self.model.score_samples(X)

    def predict(self, X):
        pred = self.model.predict(X)  # 1 = normal, -1 = anomaly
        return (pred == -1).astype(int)


class OneClassSVMDetector:
    def __init__(self, nu=0.1, kernel="rbf", gamma="scale"):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X_train, y_train=None):
        self.model.fit(X_train)
        return self

    def score(self, X):
        # decision_function: higher = more normal, so flip sign
        return -self.model.decision_function(X)

    def predict(self, X):
        pred = self.model.predict(X)  # 1 = normal, -1 = anomaly
        return (pred == -1).astype(int)


class RandomForestDetector:
    def __init__(self, n_estimators=300, max_depth=None, random_state=42,
                 class_weight="balanced"):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.score(X) >= threshold).astype(int)


class XGBDetector:
    def __init__(self, n_estimators=300, max_depth=5, learning_rate=0.05,
                 random_state=42, scale_pos_weight=None):
        if not HAS_XGB:
            raise ImportError("xgboost is not installed: pip install xgboost")
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def fit(self, X_train, y_train):
        if self.model.get_params().get("scale_pos_weight") is None:
            n_pos = max(int(y_train.sum()), 1)
            n_neg = len(y_train) - n_pos
            self.model.set_params(scale_pos_weight=n_neg / n_pos)
        self.model.fit(X_train, y_train)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.score(X) >= threshold).astype(int)


def get_all_models(contamination_estimate=0.1):
    """Convenience factory returning a dict of {name: model_instance}."""
    models = {
        "IsolationForest": IsoForestDetector(contamination=contamination_estimate),
        "OneClassSVM": OneClassSVMDetector(nu=contamination_estimate),
        "RandomForest": RandomForestDetector(),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBDetector()
    return models
