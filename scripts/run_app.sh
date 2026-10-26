#!/usr/bin/env bash
# Run the Streamlit app from the repository root:  bash scripts/run_app.sh
# Model, scaler and access-point config are resolved relative to the repository
# (override with IPS_MODEL_PATH / IPS_SCALER_PATH / IPS_AP_CONFIG if needed).
set -euo pipefail
streamlit run app/app.py
