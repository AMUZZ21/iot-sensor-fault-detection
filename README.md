# ML-Based IoT Sensor Fault / Anomaly Detection

Detects abnormal or faulty behavior (drift, spikes, noise, stuck readings) in
multivariate IoT sensor data, using the UCI Gas Sensor Array Drift Dataset,
comparing Isolation Forest, One-Class SVM, Random Forest, and XGBoost.

## 1. The most important thing to understand about the dataset

The UCI Gas Sensor Array Drift Dataset's label is **which of 6 gases was
presented to the sensor array**, not "sensor is faulty / not faulty." There is
no native fault ground truth. This is normal for this kind of project, and the
literature handles it in two complementary ways, both implemented here:

1. **Unsupervised anomaly detection** (Isolation Forest, One-Class SVM):
   train on data assumed mostly-normal, flag rare/extreme points. The
   batch-to-batch drift (12 recording sessions over ~3 years) is itself a
   realistic stress-test: sensors 8-10 months apart look statistically
   different even when nothing is "broken," and a good detector needs to
   not raise constant false alarms on drift alone while still catching real
   faults.
2. **Synthetic fault injection** (`src/fault_injection.py`): to get real
   Precision/Recall/F1/ROC-AUC numbers and to train Random Forest / XGBoost
   supervised, we programmatically corrupt a fraction of rows with four
   fault types — **spike, drift, noise, stuck-at** — on a randomly chosen
   sensor, and label those rows `1`. This is standard practice in
   sensor-fault-detection papers when no natively fault-labeled dataset
   exists, and it also gives you the labels needed for the project's
   stretch goal (which sensor caused the anomaly, and which fault type it
   was).

If you get access to the NASA sensor fault dataset and it has real fault
labels, you can swap `fault_injection.py` out for those real labels and reuse
every other module unchanged.

## 2. Project structure

```
iot_fault_detection/
├── data/
│   └── raw/              <- put batch1.dat ... batch10.dat here (see below)
├── src/
│   ├── data_loader.py         # parses UCI batch files + synthetic demo generator
│   ├── fault_injection.py     # injects spike/drift/noise/stuck faults + labels
│   ├── preprocessing.py       # cleaning, batch-based time-split, scaling
│   ├── feature_engineering.py # per-sensor stats, cross-sensor disagreement
│   ├── models.py              # IsolationForest / OneClassSVM / RF / XGBoost wrappers
│   ├── evaluate.py            # metrics, plots, fault-type breakdown, sensor localization
│   └── main.py                # end-to-end pipeline (run this)
├── notebooks/
│   └── exploration.ipynb      # EDA scaffold (optional, for your report/viva)
├── results/                   # metrics CSVs + plots get written here
├── requirements.txt
└── README.md
```

## 3. Getting the real dataset

The sandbox that generated this project can't reach `archive.ics.uci.edu`
directly, so download it yourself:

1. Go to https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
2. Download `Dataset.zip`, unzip it — you'll get `batch1.dat` ... `batch10.dat`
3. Put them in `iot_fault_detection/data/raw/`
4. Run `python src/main.py` (it auto-detects the real files; without them it
   falls back to a synthetic demo dataset that mimics the same 128-feature,
   16-sensor, 10-batch structure so you can test the whole pipeline today).

For the NASA dataset, inspect its actual file format after downloading —
you'll need a small loader analogous to `load_batch()` in `data_loader.py`;
everything downstream (fault injection, features, models, evaluation) is
dataset-agnostic as long as you end up with a DataFrame of numeric sensor
feature columns.

## 4. Running it

```bash
pip install -r requirements.txt
cd iot_fault_detection
python src/main.py                 # uses real data if present in data/raw/
python src/main.py --use-synthetic # force the synthetic demo dataset
```

Outputs land in `results/`:
- `metrics_comparison.csv` — Precision/Recall/F1/ROC-AUC/PR-AUC/False Alarm
  Rate for all four models
- `recall_by_fault_type.csv` — recall broken down by spike/drift/noise/stuck
- `roc_pr_curves.png`, `metric_comparison.png`

## 5. Methodology summary (for your report / proposal writeup)

**Preprocessing**: drop duplicates/NaNs, split **by batch (time)** rather
than randomly — training on early batches and testing on later ones is the
realistic deployment scenario for a drift dataset, and avoids the model
"seeing the future" during training. Features are standardized (z-score)
using statistics fit on the training split only.

**Feature engineering**: beyond the 128 raw steady-state/transient
descriptors (16 sensors × 8 descriptors), we add per-sensor mean/std, a
cross-sensor "disagreement" score (how far one sensor's summary is from the
consensus of the other 15, robust-scaled via median/MAD), and a per-row
count of features more than 3 std from their column mean. These make both
"this whole row looks off" and "this one sensor looks off relative to its
peers" learnable.

**Models**:
- *Isolation Forest* — unsupervised, isolates anomalies via random partitioning; fast, no distributional assumptions.
- *One-Class SVM* — unsupervised, learns a boundary around normal data in kernel space; more sensitive to scaling and contamination assumptions.
- *Random Forest* — supervised, trained on the fault-injected labels; strong baseline, gives feature importances.
- *XGBoost* — supervised, typically best raw performance on tabular data given enough labeled (here, semi-synthetic) examples; class-imbalance handled via `scale_pos_weight`.

**Evaluation**: Precision, Recall, F1, ROC-AUC, PR-AUC (more informative than
ROC-AUC under class imbalance, which this problem has), and **False Alarm
Rate** (FP / (FP+TN)) since in a real deployment, alerting on healthy sensors
too often is what gets a monitoring system ignored/disabled. We also report
recall **per fault type**, since a system might excel at catching spikes but
miss slow drift, which matters practically.

**Fault localization (stretch goal)**: for correctly-flagged anomalies, we
check whether the sensor with the highest per-row "disagreement" score
matches the sensor we actually corrupted (top-1 localization accuracy). You
can extend this using SHAP values on the Random Forest/XGBoost models for a
model-driven (rather than heuristic) localization signal — see Section 6.

## 6. Natural next steps / extensions

- Add SHAP (`pip install shap`) to explain which of the 128+engineered
  features drove each model's anomaly score — stronger fault localization
  than the disagreement heuristic, and good material for a project report.
- Try an Autoencoder (reconstruction-error anomaly score) as a 5th model —
  natural fit since you already have PyTorch/TensorFlow-style tabular data.
- Sweep `fault_fraction` and per-fault-type magnitude in `fault_injection.py`
  to build a robustness curve (metric vs. fault severity).
- If you get the NASA dataset working, repeat the same pipeline on it and
  compare cross-dataset generalization — a good "future work" section.
