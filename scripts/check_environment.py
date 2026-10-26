"""Sanity-check the local environment before launching the app.

Does NOT modify any project code. It only verifies that:
  1. the required Python packages are importable,
  2. the trained model + scaler exist and can be loaded,
  3. the private access-point config is present (otherwise the public example is used),
  4. a Wi-Fi interface is visible (optional, needed only for live scans).

Usage (from repository root):
    python scripts/check_environment.py
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.getenv("IPS_MODEL_PATH", os.path.join(ROOT, "models", "knn_model_radiomap.pkl"))
SCALER_PATH = os.getenv("IPS_SCALER_PATH", os.path.join(ROOT, "models", "scaler_radiomap.pkl"))
AP_PRIVATE = os.path.join(ROOT, "config", "access_points.private.json")
PACKAGES = ["streamlit", "numpy", "pandas", "sklearn", "joblib", "pywifi"]


def ok(msg):
    print(f"[ OK ] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def main() -> int:
    errors = 0

    print(f"Python {sys.version.split()[0]}\n")
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            ok(f"{name} {getattr(module, '__version__', '')}".strip())
        except Exception as exc:
            fail(f"{name} not importable ({exc})")
            errors += 1

    print()
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            import joblib

            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            ok(f"model  : {type(model).__name__} ({MODEL_PATH})")
            ok(f"scaler : {type(scaler).__name__} ({SCALER_PATH})")
            n_model = getattr(model, "n_features_in_", None)
            n_scaler = getattr(scaler, "n_features_in_", None)
            if n_model is not None and n_model == n_scaler:
                ok(f"feature count matches ({n_model} access points)")
            else:
                fail(f"feature count mismatch (model={n_model}, scaler={n_scaler})")
                errors += 1
        except Exception as exc:
            fail(f"could not load model/scaler: {exc}")
            errors += 1
    else:
        fail(f"model/scaler not found at {MODEL_PATH} and {SCALER_PATH} "
             "(run notebooks/main.ipynb section 7 and copy the .pkl files into models/)")
        errors += 1

    print()
    if os.path.exists(AP_PRIVATE):
        ok("private access-point config found (config/access_points.private.json)")
    else:
        warn("private access-point config missing - the placeholder example config will be used "
             "(see docs/PRIVATE_DATA.md)")

    print()
    try:
        import pywifi

        count = len(pywifi.PyWiFi().interfaces())
        if count:
            ok(f"{count} Wi-Fi interface(s) detected")
        else:
            warn("no Wi-Fi interface detected (live scanning will be disabled)")
    except Exception as exc:
        warn(f"Wi-Fi check skipped ({exc})")

    print("\nResult:", "ready" if errors == 0 else f"{errors} problem(s) found")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
