from pathlib import Path
import sys
import json
import traceback

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request, send_from_directory

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"

MODEL_PATH = BASE_DIR / "models" / "RandomForest_bundle.joblib"
RESULTS_DIR = BASE_DIR / "results 0.15"

SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static"
)


# ============================================================
# MODEL LOADING
# ============================================================

BUNDLE = None


def load_bundle():
    global BUNDLE

    if BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Saved model not found: {MODEL_PATH}"
            )

        BUNDLE = joblib.load(MODEL_PATH)

    return BUNDLE


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """Convert numpy values into JSON-safe Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def dataframe_records(df):
    """Return JSON-safe dataframe records."""
    return [
        {
            str(k): clean_value(v)
            for k, v in row.items()
        }
        for row in df.to_dict(orient="records")
    ]


def read_result_csv(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        return None

    df = pd.read_csv(path)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_raw_sample(raw_values, raw_feature_columns):
    """
    Uses the project's existing feature-engineering module.

    We deliberately try common function names/signatures so this
    web layer does not duplicate or alter the project's ML pipeline.
    """

    try:
        import feature_engineering
    except Exception as exc:
        raise RuntimeError(
            "Could not import src/feature_engineering.py"
        ) from exc

    raw_array = np.asarray(raw_values, dtype=float).reshape(1, -1)

    # DataFrame is the safest representation because the original
    # project works with named sensor channels.
    raw_df = pd.DataFrame(
        raw_array,
        columns=raw_feature_columns
    )

    candidate_names = [
        "engineer_features",
        "extract_features",
        "build_features",
        "create_features",
        "generate_features",
        "feature_engineering",
        "transform_features",
    ]

    candidates = []

    for name in candidate_names:
        fn = getattr(feature_engineering, name, None)

        if callable(fn):
            candidates.append(fn)

    # Also inspect any callable in the module if the expected name
    # is different.
    if not candidates:
        for name in dir(feature_engineering):
            if name.startswith("_"):
                continue

            obj = getattr(feature_engineering, name)

            if callable(obj):
                candidates.append(obj)

    errors = []

    for fn in candidates:
        attempts = [
            lambda: fn(raw_df),
            lambda: fn(raw_df, raw_feature_columns),
            lambda: fn(raw_array),
        ]

        for attempt in attempts:
            try:
                result = attempt()

                if isinstance(result, pd.DataFrame):
                    feature_df = result.copy()

                elif isinstance(result, pd.Series):
                    feature_df = result.to_frame().T

                else:
                    arr = np.asarray(result)

                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)

                    feature_df = pd.DataFrame(arr)

                if feature_df.shape[0] != 1:
                    continue

                return feature_df

            except Exception as exc:
                errors.append(str(exc))

    raise RuntimeError(
        "Could not reproduce the project's feature-engineering "
        "function automatically.\n"
        + "\n".join(errors[-5:])
    )


def prepare_model_input(raw_values, bundle):
    """
    Recreate the same feature matrix expected by the saved model.
    """

    raw_feature_columns = bundle.get("raw_feature_columns")

    if raw_feature_columns is None:
        raise RuntimeError(
            "Saved bundle does not contain raw_feature_columns."
        )

    if len(raw_values) != len(raw_feature_columns):
        raise ValueError(
            f"Expected {len(raw_feature_columns)} raw sensor values, "
            f"received {len(raw_values)}."
        )

    feature_df = engineer_raw_sample(
        raw_values,
        raw_feature_columns
    )

    expected_features = bundle.get("feature_columns")

    if expected_features is not None:

        # If the feature-engineering output has names, align them.
        if all(
            column in feature_df.columns
            for column in expected_features
        ):
            feature_df = feature_df[expected_features]

        elif feature_df.shape[1] == len(expected_features):
            feature_df.columns = expected_features

        else:
            raise RuntimeError(
                f"Feature-engineering produced "
                f"{feature_df.shape[1]} features, but the model "
                f"expects {len(expected_features)}."
            )

    return feature_df


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_from_raw(raw_values):
    bundle = load_bundle()

    model = bundle["model"]
    scaler = bundle["scaler"]

    X = prepare_model_input(
        raw_values,
        bundle
    )

    X_scaled = scaler.transform(X)

    # Support either the project's wrapper model or sklearn directly.
    predictor = model

    if hasattr(predictor, "predict"):
        prediction = predictor.predict(X_scaled)

    elif hasattr(predictor, "model"):
        prediction = predictor.model.predict(X_scaled)

    else:
        raise RuntimeError(
            "Saved model does not expose predict()."
        )

    prediction_value = int(np.asarray(prediction).reshape(-1)[0])

    probability = None

    if hasattr(predictor, "predict_proba"):
        probability = predictor.predict_proba(X_scaled)[0]

    elif hasattr(predictor, "model") and hasattr(
        predictor.model,
        "predict_proba"
    ):
        probability = predictor.model.predict_proba(X_scaled)[0]

    result = {
        "prediction": prediction_value,
        "label": "FAULT DETECTED" if prediction_value == 1 else "NORMAL",
        "status": "fault" if prediction_value == 1 else "normal",
        "features_used": int(X.shape[1]),
    }

    if probability is not None:
        classes = getattr(
            predictor,
            "classes_",
            getattr(
                getattr(predictor, "model", None),
                "classes_",
                [0, 1]
            )
        )

        probabilities = {}

        for cls, prob in zip(classes, probability):
            probabilities[str(int(cls))] = float(prob)

        result["probabilities"] = probabilities

        if "1" in probabilities:
            result["fault_probability"] = probabilities["1"]

        if "0" in probabilities:
            result["normal_probability"] = probabilities["0"]

    return result


# ============================================================
# DEMO SAMPLE GENERATOR
# ============================================================

def make_demo_sample(strength=0.0):
    """
    Creates a deterministic presentation sample from the training
    distribution stored inside the model bundle.

    strength = number of training standard deviations applied to
    the first group of sensor channels.
    """

    bundle = load_bundle()

    mean = np.asarray(
        bundle["raw_train_mean"],
        dtype=float
    ).copy()

    std = np.asarray(
        bundle["raw_train_std"],
        dtype=float
    ).copy()

    # Avoid zero standard deviations.
    std[std == 0] = 1.0

    sample = mean.copy()

    if strength != 0:
        # Strong, controlled spike affecting several channels.
        affected = min(8, len(sample))

        sample[:affected] = (
            mean[:affected]
            + float(strength) * std[:affected]
        )

    return sample


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def index():
    return send_from_directory(
        STATIC_DIR,
        "index.html"
    )


@app.get("/api/status")
def api_status():

    try:
        bundle = load_bundle()

        return jsonify({
            "ok": True,
            "model": "Random Forest",
            "model_file": str(MODEL_PATH.name),
            "results_folder": RESULTS_DIR.name,
            "features": len(
                bundle.get("feature_columns", [])
            ),
            "raw_features": len(
                bundle.get("raw_feature_columns", [])
            ),
            "train_batches": bundle.get(
                "train_batches",
                []
            ),
            "test_batches": bundle.get(
                "test_batches",
                []
            ),
            "fault_fraction": bundle.get(
                "fault_fraction",
                0.15
            )
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500


@app.get("/api/results")
def api_results():

    try:
        df = read_result_csv(
            "metrics_comparison.csv"
        )

        if df is None:
            return jsonify({
                "ok": False,
                "error": "metrics_comparison.csv not found."
            }), 404

        return jsonify({
            "ok": True,
            "columns": [
                str(c) for c in df.columns
            ],
            "rows": dataframe_records(df)
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500


@app.get("/api/fault-recall")
def api_fault_recall():

    try:
        df = read_result_csv(
            "recall_by_fault_type.csv"
        )

        if df is None:
            return jsonify({
                "ok": False,
                "error": "recall_by_fault_type.csv not found."
            }), 404

        return jsonify({
            "ok": True,
            "columns": [
                str(c) for c in df.columns
            ],
            "rows": dataframe_records(df)
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500


@app.get("/results/<path:filename>")
def result_asset(filename):

    return send_from_directory(
        RESULTS_DIR,
        filename
    )


@app.post("/api/demo")
def api_demo():

    try:
        payload = request.get_json(
            silent=True
        ) or {}

        strength = float(
            payload.get(
                "strength",
                0
            )
        )

        raw_sample = make_demo_sample(
            strength
        )

        result = predict_from_raw(
            raw_sample
        )

        result["demo_strength"] = strength
        result["sample_type"] = (
            "baseline"
            if strength == 0
            else f"{strength:g}σ controlled spike"
        )

        return jsonify({
            "ok": True,
            **result
        })

    except Exception as exc:

        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500


@app.post("/api/predict")
def api_predict():

    try:

        payload = request.get_json(
            silent=True
        ) or {}

        values = payload.get(
            "values"
        )

        if values is None:
            return jsonify({
                "ok": False,
                "error": "No sensor values supplied."
            }), 400

        values = [
            float(v)
            for v in values
        ]

        result = predict_from_raw(
            values
        )

        return jsonify({
            "ok": True,
            **result
        })

    except Exception as exc:

        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("  IS THIS SENSOR LYING?")
    print("  IoT Sensor Fault Detection Dashboard")
    print("=" * 60)
    print()

    try:
        bundle = load_bundle()

        print("Model loaded:")
        print(f"  {MODEL_PATH}")

        print(
            f"Features: "
            f"{len(bundle.get('feature_columns', []))}"
        )

        print(
            f"Raw sensors: "
            f"{len(bundle.get('raw_feature_columns', []))}"
        )

        print(
            f"Fault fraction: "
            f"{bundle.get('fault_fraction')}"
        )

    except Exception as exc:

        print(
            "WARNING: model could not be loaded:"
        )

        print(exc)

    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
