# System Architecture

## Components

| Component | File | Role |
|---|---|---|
| Data pipeline | `notebooks/main.ipynb` (sections 2-4) | Reads each device's CSVs, filters by SSID keyword, normalizes MAC addresses, builds the radio map |
| Experiments & validation | `notebooks/main.ipynb` (sections 5-6) | K tuning, confusion matrices, Group K-Fold, LOGO |
| Final training | `notebooks/main.ipynb` (section 7) | Fits scaler + KNN on the full data and saves them with `joblib` |
| Inference app | `app/app.py` | Scans Wi-Fi, builds the feature vector, scales it, predicts, displays the result |
| Site configuration | `config/access_points.*.json` | SSID keyword, site description, and the AP table (BSSID -> label) |

## Training Data Flow

1. Each CSV is named `<location>_<point>.csv` and stored in `data/raw/device_1` ... `device_4`.
2. Columns are mapped per device (see [`DATASET.md`](DATASET.md)).
3. MAC addresses are normalized (lowercase, `-` becomes `:`), and only the APs in the configured whitelist are kept.
4. APs that were not captured are filled with `-100 dBm`.
5. The result is saved as `data/processed/processed_fingerprint_data.csv`.

## Inference Flow (App)

1. The user presses **Scan access points**.
2. `pywifi` triggers a scan and the app waits `SCAN_DURATION` (12 s) for the adapter to finish.
3. Each target AP's RSSI is matched by BSSID; APs that are not visible get `-100`.
4. The feature vector is ordered like `TARGET_APS`, then scaled with the training scaler.
5. KNN predicts the location; confidence is the maximum value of `predict_proba`.

## AP Order Matters

The feature order must be **identical** between the notebook (`list_ap`) and the app (`TARGET_APS`).
Both are generated from the same `access_points` table in the config file, so they stay in sync.
Changing the order or the number of APs means the model must be retrained.

## Design Decisions

- **KNN + Manhattan + distance weighting**: simple, no heavy training, well suited to a small radio map.
- **Group-based validation**: splitting by reference point prevents leakage between neighbouring points.
- **Server-side scanning**: `pywifi` talks to the Wi-Fi adapter of the machine running Streamlit.
- **Config-driven site data**: nothing site-specific is hard-coded, so the repository can be public.
