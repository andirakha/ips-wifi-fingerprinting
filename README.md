# 📡 Indoor Positioning Lab — Wi-Fi Fingerprinting with KNN

A prototype **Indoor Positioning System (IPS)** based on *Wi-Fi RSSI fingerprinting*.
It scans nearby access points (APs), builds an RSSI feature vector from 12 target APs, and predicts the
room a device is in using a **K-Nearest Neighbors (KNN)** classifier.

> Developed as part of a bachelor's thesis project. This is a research prototype, not a production system.

## ✨ Features

- **Data & modelling pipeline** (`notebooks/main.ipynb`): multi-device CSV parsing, radio-map preprocessing,
  standardization, K tuning, *Group K-Fold* and *Leave-One-Group-Out (LOGO)* validation, and model export.
- **Live scanner app** (`app/app.py`): a Streamlit interface that scans Wi-Fi with `pywifi`, shows the signal
  strength of every target AP, and reports the predicted position, confidence, and inference time.
- **Complete project environment**: split requirements (app / notebook / dev), run scripts, an environment
  check, a sensitive-content scanner with a pre-commit hook, GitHub Actions CI, and documentation.
- **Private data kept out of Git**: site-specific values live in git-ignored files (see
  [`docs/PRIVATE_DATA.md`](docs/PRIVATE_DATA.md)).

## 🧭 System Flow

```
Wi-Fi scan CSVs ──► Parsing & preprocessing ──► Radio map (12 APs x locations)
 (4 devices)            (notebook)                         │
                                                           ▼
                                            StandardScaler + KNN (K=11)
                                            (weights=distance, manhattan)
                                                           │  joblib
                                                           ▼
Wi-Fi adapter ──► pywifi scan ──► RSSI vector ──► scaler ──► KNN ──► location + confidence
                                  (app/app.py, Streamlit)
```

More detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 📊 Results (summary)

| Evaluation | Mean accuracy |
|---|---|
| Manual split (points 1-5 train, 6-9 validation), K=11 | 92.5% |
| Group K-Fold (5 folds) | 87.5% |
| Leave-One-Group-Out (LOGO) | 88.6% |

Dataset: 360 fingerprints, 10 locations, 9 reference points per location, 4 devices, 12 target APs.
Full breakdown in [`docs/RESULTS.md`](docs/RESULTS.md).

## 🗂️ Repository Structure

```
ips-wifi-fingerprinting/
├── app/
│   └── app.py                      # Streamlit app (live scanner + prediction)
├── notebooks/
│   └── main.ipynb                  # Preprocessing, validation, training (outputs cleared)
├── config/
│   ├── access_points.example.json  # Public placeholder AP table
│   └── access_points.private.json  # REAL AP table (git-ignored, you supply it)
├── data/
│   ├── raw/                        # Per-device scan CSVs (git-ignored)
│   └── processed/                  # Preprocessed datasets (git-ignored)
├── models/                         # knn_model_radiomap.pkl, scaler_radiomap.pkl (git-ignored)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── PRIVATE_DATA.md
│   ├── RESULTS.md
│   └── SETUP.md
├── scripts/
│   ├── check_environment.py        # Checks packages, model, scaler, Wi-Fi adapter
│   ├── check_sensitive.py          # Scans for BSSIDs, personal paths, notebook outputs
│   ├── run_app.ps1                 # Run the app (Windows)
│   └── run_app.sh                  # Run the app (Linux/macOS)
├── .githooks/pre-commit            # Blocks commits containing sensitive content
├── .github/workflows/ci.yml        # Lint + compile + notebook check + sensitive scan
├── .streamlit/config.toml
├── .env.example
├── requirements*.txt
├── Makefile
├── pyproject.toml
└── LICENSE
```

## 🚀 Quick Start

```bash
# 1. Clone and enter the repository
git clone https://github.com/<username>/ips-wifi-fingerprinting.git
cd ips-wifi-fingerprinting

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Enable the sensitive-content guard (once per clone)
git config core.hooksPath .githooks

# 5. Add your private files (see docs/PRIVATE_DATA.md), then verify the setup
python scripts/check_environment.py

# 6. Launch the app
.\scripts\run_app.ps1             # Windows
# bash scripts/run_app.sh         # Linux/macOS
```

To run the notebook: `pip install -r requirements-notebook.txt`, then `jupyter lab notebooks/`.
Without private files the app and notebook fall back to the placeholder config in `config/access_points.example.json`.

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `IPS_MODEL_PATH` | `<repo>/models/knn_model_radiomap.pkl` | KNN model file |
| `IPS_SCALER_PATH` | `<repo>/models/scaler_radiomap.pkl` | StandardScaler file |
| `IPS_AP_CONFIG` | `<repo>/config/access_points.private.json` | AP table (falls back to the example file) |

## 🧪 Method Summary

| Component | Setting |
|---|---|
| Features | RSSI (dBm) of 12 target APs (2.4 GHz and 5 GHz); `-100` when an AP is not detected |
| Scaler | `StandardScaler` |
| Model | `KNeighborsClassifier(n_neighbors=11, weights='distance', metric='manhattan')` |
| Validation | Manual split, Group K-Fold (5 folds), LOGO |
| Label | Room / location name |

## 🔒 Sensitive Data Policy

Real BSSIDs, SSID keyword, site description, raw scans, trained models, and notebook outputs are **never
committed**. They are split into git-ignored files and restored locally on demand. `scripts/check_sensitive.py`
(run by the pre-commit hook and by CI) fails if anything sensitive slips into a commit.
Full guide: [`docs/PRIVATE_DATA.md`](docs/PRIVATE_DATA.md).

## ⚠️ Notes & Limitations

- The model is trained for a specific set of APs and environment; **retrain** it for a different building.
- Position estimates depend on signal conditions, device hardware, and the environment.
- Wi-Fi scanning runs on the **server device** (not in the browser), so the app must run on a machine with a
  Wi-Fi adapter (tested on Windows).
- The `.pkl` files must be loaded with the same `scikit-learn` version used to create them.

## 🛣️ Roadmap

- [ ] Compare other algorithms (Random Forest, SVM) on the same radio map
- [ ] Log prediction history to CSV
- [ ] Floor-plan visualization of the predicted position
- [ ] Unit tests for the preprocessing steps

## 📄 License

[MIT](LICENSE)
