# Setup Guide

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Windows with a Wi-Fi adapter for live scanning. The notebook runs on any OS.

## 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt            # app only
pip install -r requirements-notebook.txt   # app + notebook
git config core.hooksPath .githooks        # enable the sensitive-content guard
```

## 2. Private Files

Copy your private files into place (full details in [`PRIVATE_DATA.md`](PRIVATE_DATA.md)):

- `config/access_points.private.json` - your real AP table
- `data/raw/device_*/...` - your raw scan CSVs (only needed to retrain)

## 3. Prepare the Model

The model is not committed. Either:

1. **Train it yourself**: put the data in `data/raw/`, run all of `notebooks/main.ipynb`. Section 7 writes
   `models/knn_model_radiomap.pkl` and `models/scaler_radiomap.pkl` automatically (paths are repo-relative).
2. **Download from GitHub Releases** (if you attach the files there) and place them in `models/`.

## 4. Verify

```bash
python scripts/check_environment.py
python scripts/check_sensitive.py
```

## 5. Run the App

```powershell
.\scripts\run_app.ps1
```

or

```bash
bash scripts/run_app.sh
```

Open `http://localhost:8501`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model unavailable` in the top-right pill | The `.pkl` files are missing from `models/`, or the path is wrong |
| `Wi-Fi adapter not detected` | Run on Windows with an active Wi-Fi adapter; try `pip install comtypes` |
| Error in `joblib.load` | `scikit-learn` version differs from training; match it or retrain |
| Every RSSI is `-100` | Make sure you are in range of the target APs and that `config/access_points.private.json` holds the right BSSIDs |
| App shows placeholder AP names | The private config is missing, so the example config is in use |
