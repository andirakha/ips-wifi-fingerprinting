import html
import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pywifi
import streamlit as st


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repository root (app/ lives one level below)
MODEL_PATH = os.getenv("IPS_MODEL_PATH", str(PROJECT_ROOT / "models" / "knn_model_radiomap.pkl"))
SCALER_PATH = os.getenv("IPS_SCALER_PATH", str(PROJECT_ROOT / "models" / "scaler_radiomap.pkl"))
SCAN_DURATION = 12  # seconds, wait time so the Wi-Fi adapter can finish scanning

st.set_page_config(page_title="Indoor Positioning Lab", page_icon="📡", layout="wide")


# --------------------------------------------------------------------------
# Styling (port of src/index.css + Tailwind classes from App.tsx)
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

/* --- Streamlit chrome --- */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
.stApp { background: #f3f5f2; color: #14251d; font-family: 'Manrope', sans-serif; overflow-x: hidden; }
.block-container { max-width: 1240px; padding: 0 24px 48px !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stHorizontalBlock"] { gap: 24px !important; }
.stMarkdown p { margin: 0; }
.stApp,
[data-testid="stMain"],
[data-testid="stAppViewContainer"],
section.main {
  overflow-x: hidden !important;
}

/* --- Design tokens --- */
.ips * { box-sizing: border-box; }
.ips { font-family: 'Manrope', sans-serif; color: #14251d; }
.ips .display { font-family: 'Newsreader', serif; }
.ips .mono { font-family: 'DM Mono', monospace; }
.ips .eyebrow { color: #a24c2b; font-size: 10px !important; font-weight: 700; letter-spacing: .18em; margin: 0 !important; }
.ips p, .ips h1, .ips h3, .ips dl, .ips dd { margin: 0; }
.ips .panel { border: 1px solid #dfe4df; border-radius: 16px; background: #fff; padding: 20px;
              box-shadow: 0 1px 2px rgb(20 37 29 / .03); }

/* --- Top bar (full bleed) --- */
.ips-topbar { width: 100vw; margin-left: calc(50% - 50vw) !important; border-bottom: 1px solid #d8ddd8; background: #f8faf7; }
.ips-topbar-inner { max-width: 1240px; margin: 0 auto !important; padding: 16px 24px; display: flex;
                    align-items: center; justify-content: space-between; }
.ips-brand { display: flex; align-items: center; gap: 12px; }
.ips-logo { width: 40px; height: 40px; border-radius: 12px; background: #183c2d; color: #fff;
            display: grid; place-items: center; font-size: 14px !important; font-weight: 700; }
.ips-brand-title { font-size: 18px !important; font-weight: 600; line-height: 28px; }
.ips-brand-sub { font-size: 12px !important; color: #718078; line-height: 16px; }
.ips-pill { display: flex; align-items: center; gap: 8px; border: 1px solid #cad4cd; background: #fff;
            border-radius: 999px; padding: 6px 12px; font-size: 12px !important; font-weight: 500; }
.ips-dot { width: 8px; height: 8px; border-radius: 999px; background: #2ca66f; }
.ips-dot.off { background: #c0563a; }

/* --- Hero --- */
.ips-hero { padding-top: 40px; padding-bottom: 16px; }
.ips-hero h1 { margin-top: 12px; font-size: 48px !important; line-height: 1; font-weight: 600; letter-spacing: -.03em;
                max-width: 672px; padding: 0 !important; }
.ips-hero .lead { margin-top: 16px !important; max-width: 672px; font-size: 14px !important; line-height: 24px; color: #69766f; }
.ips-rule { border-bottom: 1px solid #d9dfda; margin-top: 32px !important; }

/* --- Scan button --- */
.stButton > button,
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  width: 100%; min-width: 208px; border-radius: 12px;
  background-color: #183c2d !important; color: #fff !important;
  border: 0 !important; padding: 14px 20px;
  font-family: 'Manrope', sans-serif; font-size: 14px !important;
  font-weight: 600; box-shadow: 0 1px 2px rgb(0 0 0 / .05);
  transition: background-color .15s;
}
.stButton > button:hover,
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
  background-color: #25533e !important; color: #fff !important; border: 0 !important;
}
.stButton > button:active,
[data-testid="stBaseButton-primary"]:active {
  background-color: #102a1f !important; color: #fff !important;
}
.stButton > button:focus:not(:active),
[data-testid="stBaseButton-primary"]:focus:not(:active) {
  background-color: #183c2d !important; color: #fff !important; border: 0 !important;
  box-shadow: 0 0 0 2px #a24c2b55;
}
.stButton > button p { font-weight: 600; color: #fff !important; }

/* --- Scan button: disabled (not ready / scanning) --- */
.stButton > button:disabled,
.stButton > button[kind="primary"]:disabled,
[data-testid="stBaseButton-primary"]:disabled {
  background-color: #e4e9e5 !important; color: #85928a !important;
  box-shadow: inset 0 0 0 1px #d2d9d3 !important; cursor: not-allowed !important;
}
.stButton > button:disabled:hover,
[data-testid="stBaseButton-primary"]:disabled:hover { background-color: #e4e9e5 !important; }
.stButton > button:disabled p { color: #85928a !important; }
.st-key-scan_busy button:disabled { cursor: progress !important; }
.st-key-scan_busy button:disabled p { color: #4d6558 !important; }
.st-key-scan_busy button::before {
  content: ""; width: 14px; height: 14px; margin-right: 10px; flex: none;
  border: 2px solid rgb(24 60 45 / .2); border-top-color: #183c2d; border-radius: 50%;
  animation: ips-spin .8s linear infinite;
}
@keyframes ips-spin { to { transform: rotate(360deg); } }

/* --- Main grid --- */
.ips-grid { display: grid; grid-template-columns: 1fr 1.65fr; gap: 24px; margin-top: 48px !important; height: 531px; }
.ips-table-panel { display: flex; flex-direction: column; min-height: 0;
                   padding: 0 20px 20px !important; overflow: hidden; }
.ips-table-head { display: flex; align-items: center; justify-content: space-between;
                  padding: 16px 0; gap: 12px; }
.ips-table-head .t { font-weight: 600; font-size: 16px !important; }
.ips-table-head .s { margin-top: 4px !important; font-size: 12px !important; color: #7b8881; }
.ips-badge { border-radius: 999px; background: #edf2ee; padding: 4px 12px; font-size: 12px !important; font-weight: 600;
              text-align: center; }
.ips-table-scroll { flex: 1; min-height: 0; overflow: auto; }
.ips table { width: 100%; border-collapse: collapse; text-align: left; font-size: 14px !important; }
.ips th, .ips td { padding: 14px 18px; border-left: 0; border-right: 0; white-space: nowrap; }
.ips thead { position: sticky; top: 0; z-index: 1; }
.ips thead th { position: sticky; top: 0; z-index: 2; background: #f7f9f7;
                box-shadow: inset 0 1px 0 #e2e6e2, inset 0 -1px 0 #e2e6e2; }
.ips thead tr { background: #f7f9f7; font-size: 10px !important; text-transform: uppercase; letter-spacing: .14em; color: #7b8881; }
.ips th { font-weight: 600; }
.ips tbody tr { border-top: 1px solid #edf0ed; }
.ips tbody tr:last-child td { border-bottom: 0; }
.ips .rank { color: #8b968f; }
.ips .apname { font-weight: 600; }
.ips .bar { height: 6px; width: 128px; overflow: hidden; border-radius: 999px; background: #e6ebe7; }
.ips .bar > div { height: 100%; border-radius: 999px; background: #527462; }
.ips .rssi { font-weight: 500; color: #a24c2b; }

/* --- Aside --- */
.ips-aside { display: flex; flex-direction: column; gap: 20px; min-height: 0; }
.ips-result { border-radius: 16px; background: #183c2d; color: #fff; padding: 24px;
              box-shadow: 0 1px 2px rgb(0 0 0 / .05); }
.ips-result .lbl { font-size: 10px !important; font-weight: 600; letter-spacing: .18em; color: #a8c3b5; }
.ips-result .val { font-size: 36px !important; line-height: 40px; font-weight: 600; margin-top: 32px !important; }
.ips-result .sub { margin-top: 4px !important; font-size: 14px !important; color: #b9cbc2; }
.ips-result .stats { margin-top: 32px !important; display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
                     border-top: 1px solid rgb(255 255 255 / .15); padding-top: 20px; }
.ips-result .stats .k { font-size: 12px !important; color: #a8bbb1; }
.ips-result .stats .v { font-size: 20px !important; margin-top: 4px !important; }
.ips-metrics { margin-top: 16px !important; font-size: 14px !important; display: flex; flex-direction: column; gap: 12px; }
.ips .metric { display: flex; justify-content: space-between; border-bottom: 1px solid #edf0ed;
               padding-bottom: 10px; color: #728078; }
.ips .metric dd { color: #273c32; font-weight: 600; }
.ips .metric dd.ok { color: #2a835b; }
.ips .metric dd.err { color: #c0563a; }

/* --- Research notes --- */
.ips-notes { margin-top: 48px !important; border-top: 1px solid #d9dfda; padding-top: 28px; }
.ips-notes-grid { margin-top: 16px !important; display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.ips-note { border-radius: 12px; border: 1px solid #dfe4df; background: #f8faf7; padding: 20px; }
.ips-note .n { font-size: 12px !important; color: #a24c2b; }
.ips-note h3 { margin-top: 12px !important; font-size: 16px !important; font-weight: 500; padding: 0 !important; }
.ips-note p { margin-top: 8px !important; font-size: 12px !important; line-height: 20px; color: #6f7c75; }

/* --- Alerts from Streamlit (scan errors) --- */
[data-testid="stAlert"] { margin-top: 16px !important; border-radius: 12px; }

@media (max-width: 900px) {
  .ips-grid { grid-template-columns: 1fr; height: auto; }
  .ips-table-panel { max-height: 480px; }
  .ips-notes-grid { grid-template-columns: 1fr; }
  .ips-hero h1 { font-size: 36px !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def compact(markup: str) -> str:
    """Collapse indentation/newlines so Markdown never treats the HTML as a code block."""
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def esc(value) -> str:
    """HTML-escape any value after converting it to a string."""
    return html.escape(str(value))


# --------------------------------------------------------------------------
# Model + Wi-Fi scanning (logic from main.py)
# --------------------------------------------------------------------------
# Target access points, filtered by MAC address (BSSID).
# The real table is private and lives in config/access_points.private.json (git-ignored).
# If it is missing, the public placeholder config/access_points.example.json is used instead.
def load_ap_config():
    """Load the access-point table from the private config, falling back to the public example."""
    private_path = Path(os.getenv("IPS_AP_CONFIG", str(PROJECT_ROOT / "config" / "access_points.private.json")))
    path = private_path if private_path.exists() else PROJECT_ROOT / "config" / "access_points.example.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


AP_CONFIG = load_ap_config()
SITE_DESCRIPTION = AP_CONFIG["site_description"]
TARGET_APS = {bssid.lower(): location for bssid, location in AP_CONFIG["access_points"].items()}
AP_COUNT = len(TARGET_APS)


@st.cache_resource(show_spinner=False)
def load_ml_components(model_path, scaler_path):
    """Load the model and scaler using joblib."""
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(
            f"File {model_path} or {scaler_path} was not found "
            "(make sure the files are in the same directory)"
        )
    return joblib.load(model_path), joblib.load(scaler_path)


def detect_wifi_adapter():
    """Return True if at least one Wi-Fi interface is detected."""
    try:
        return len(pywifi.PyWiFi().interfaces()) > 0
    except Exception:
        return False


def get_campus_networks(scan_duration_seconds):
    """Scan Wi-Fi and return the RSSI of every TARGET_APS entry (-100 if not detected)."""
    networks_dict = {
        bssid: {"bssid": bssid, "ap_location": location, "rssi": -100, "ssid": "Out of Range"}
        for bssid, location in TARGET_APS.items()
    }

    wifi = pywifi.PyWiFi()
    if len(wifi.interfaces()) == 0:
        raise RuntimeError("No Wi-Fi interface was found on this device.")

    iface = wifi.interfaces()[0]
    iface.scan()
    time.sleep(scan_duration_seconds)
    scan_results = iface.scan_results()

    for profile in scan_results:
        current_bssid = profile.bssid.lower() if profile.bssid else ""

        # Strip the string from the rightmost colon (pywifi BSSID format)
        if ":" in current_bssid:
            current_bssid = "".join(current_bssid.rsplit(":", 1))

        if current_bssid not in networks_dict:
            continue

        rssi = profile.signal  # already in dBm
        if networks_dict[current_bssid]["rssi"] < rssi:
            networks_dict[current_bssid]["rssi"] = rssi

            ssid_str = profile.ssid
            try:
                ssid_str = profile.ssid.encode("raw_unicode_escape").decode("utf-8", "ignore")
            except UnicodeDecodeError:
                try:
                    ssid_str = profile.ssid.encode("latin-1").decode("utf-8", "ignore")
                except Exception:
                    ssid_str = str(profile.ssid)
            except AttributeError:
                ssid_str = "<Hidden Network>"
            networks_dict[current_bssid]["ssid"] = ssid_str if ssid_str else "<Hidden Network>"

    return list(networks_dict.values())


def predict_location(networks_data, model, scaler):
    """Build features in TARGET_APS order, scale them, then predict with confidence."""
    network_lookup = {net["bssid"]: net["rssi"] for net in networks_data}
    feature_vector = [network_lookup.get(bssid, -100) for bssid in TARGET_APS.keys()]
    X_input = np.array(feature_vector).reshape(1, -1)

    started = time.perf_counter()
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]

    confidence = None
    try:
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(X_scaled)[0]))
    except Exception:
        pass
    elapsed = (time.perf_counter() - started) * 1000

    return {
        "location": str(prediction),
        "confidence": confidence,
        "elapsed": elapsed,
    }


try:
    model, scaler = load_ml_components(MODEL_PATH, SCALER_PATH)
    model_error = None
except Exception as exc:
    model, scaler = None, None
    model_error = str(exc)

wifi_adapter_ok = detect_wifi_adapter()


# --------------------------------------------------------------------------
# Top bar
# --------------------------------------------------------------------------
if model is not None:
    pill = f"<span class='ips-dot'></span> {esc(type(model).__name__)} active"
else:
    pill = "<span class='ips-dot off'></span> Model unavailable"

st.markdown(compact(f"""
<div class="ips">
  <div class="ips-topbar"><div class="ips-topbar-inner">
    <div class="ips-brand">
      <div class="ips-logo">IPS</div>
      <div><p class="display ips-brand-title">Indoor Positioning Lab</p>
           <p class="ips-brand-sub">Live signal workspace</p></div>
    </div>
    <div class="ips-pill">{pill}</div>
  </div></div>
</div>
"""), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Hero + scan button
# --------------------------------------------------------------------------
hero_left, hero_right = st.columns([3, 1], vertical_alignment="bottom")
with hero_left:
    st.markdown(compact("""
    <div class="ips ips-hero">
      <p class="eyebrow">INDOOR POSITIONING / LIVE SCANNER</p>
      <h1 class="display">Locate a device from nearby access points.</h1>
      <p class="lead">Scan Wi-Fi signals on the server device, use the predefined target access points as features, and automatically run position prediction.</p>
    </div>
    """), unsafe_allow_html=True)
with hero_right:
    scanning = st.session_state.get("scanning", False)
    model_ok = model is not None and scaler is not None
    scan_ready = model_ok and wifi_adapter_ok
    if scanning:
        scan_hint = "Scan in progress"
    elif not model_ok:
        scan_hint = "Model failed to load"
    elif not wifi_adapter_ok:
        scan_hint = "Wi-Fi adapter not detected"
    else:
        scan_hint = None
    scan_clicked = st.button(
        "Scanning…" if scanning else "Scan access points",
        type="primary",
        use_container_width=True,
        disabled=scanning or not scan_ready,
        help=scan_hint,
        key="scan_busy" if scanning else "scan_idle",
    )

st.markdown("<div class='ips-rule'></div>", unsafe_allow_html=True)

if scan_clicked and scan_ready:
    # Lock the button first (rerun); the scan itself runs at the end of the script
    st.session_state["scanning"] = True
    st.session_state["scan_notices"] = []
    st.rerun()

for level, message in st.session_state.get("scan_notices", []):
    (st.error if level == "error" else st.warning)(message)

scan_frame = st.session_state.get("scan")
scanned_ok = scan_frame is not None
detected = int((scan_frame["RSSI (dBm)"] > -100).sum()) if scanned_ok else 0


# --------------------------------------------------------------------------
# Access point table (the AP list is shown before any scan)
# --------------------------------------------------------------------------
if scanned_ok:
    ap_rows = list(zip(scan_frame["Access Point"], scan_frame["RSSI (dBm)"]))
    badge = f"{detected}/{len(ap_rows)} AP"
    table_sub = "Sorted from strongest to weakest signal"
else:
    ap_rows = [(location, None) for location in TARGET_APS.values()]
    badge = f"{AP_COUNT} AP"
    table_sub = "Target access points to scan"

row_html = ""
for index, (name, signal) in enumerate(ap_rows):
    if signal is None or int(signal) <= -100:
        width, rssi_text = 0, "—"
    else:
        width = max(8, min(100, (int(signal) + 100) * 1.25))
        rssi_text = f"{int(signal)} dBm"
    row_html += (
        f"<tr><td class='mono rank'>{index + 1:02d}</td>"
        f"<td class='apname'>{esc(name)}</td>"
        f"<td><div class='bar'><div style='width:{width}%'></div></div></td>"
        f"<td class='mono rssi'>{rssi_text}</td></tr>"
    )

table_html = f"""
<section class="panel ips-table-panel">
  <div class="ips-table-head">
    <div><p class="t">Detected access points</p>
         <p class="s">{table_sub}</p></div>
    <span class="ips-badge">{badge}</span>
  </div>
  <div class="ips-table-scroll">
    <table><thead><tr><th>Rank</th><th>Access point</th><th>Signal strength</th><th>RSSI</th></tr></thead>
    <tbody>{row_html}</tbody></table>
  </div>
</section>
"""


# --------------------------------------------------------------------------
# Prediction card + status
# --------------------------------------------------------------------------
pred_value, pred_sub, conf_text, inf_text = "—", "Scan to start prediction", "—", "—"
result = st.session_state.get("prediction")
if scanned_ok:
    if result is None:
        pred_sub = "Prediction failed or not available"
    else:
        pred_value = esc(result["location"])
        pred_sub = SITE_DESCRIPTION
        conf_text = f"{result['confidence']:.1%}" if result["confidence"] is not None else "N/A"
        inf_text = f"{result['elapsed']:.0f} ms"

if not model_ok:
    status_state, status_class = "Model unavailable", "err"
elif not wifi_adapter_ok:
    status_state, status_class = "Adapter not detected", "err"
elif scanning:
    status_state, status_class = "Scanning", ""
elif scanned_ok:
    status_state, status_class = "Complete", "ok"
else:
    status_state, status_class = "Ready", ""

metrics = [
    ("Model", "Loaded" if model is not None else "Unavailable", "" if model is not None else "err"),
    ("Wi-Fi adapter", "Detected" if wifi_adapter_ok else "Not detected", "" if wifi_adapter_ok else "err"),
    ("Last updated", st.session_state.get("scan_time", "Not scanned"), ""),
    ("Status", status_state, status_class),
]
metric_html = "".join(
    f"<div class='metric'><dt>{k}</dt><dd class='{c}'>{v}</dd></div>" for k, v, c in metrics
)

aside_html = f"""
<aside class="ips-aside">
  <section class="ips-result">
    <p class="lbl">PREDICTED POSITION</p>
    <p class="display val">{pred_value}</p>
    <p class="sub">{pred_sub}</p>
    <div class="stats">
      <div><p class="k">Confidence</p><p class="mono v">{conf_text}</p></div>
      <div><p class="k">Inference</p><p class="mono v">{inf_text}</p></div>
    </div>
  </section>
  <section class="panel">
    <p class="eyebrow">SCAN STATUS</p>
    <dl class="ips-metrics">{metric_html}</dl>
  </section>
</aside>
"""

st.markdown(
    compact(f"<div class='ips'><div class='ips-grid'>{aside_html}{table_html}</div></div>"),
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Research notes
# --------------------------------------------------------------------------
notes = [
    ("01", "Purpose of Use", "This project is a prototype developed for a bachelor's thesis and is not intended for production deployment."),
    ("02", "Model Scope", "The model is trained on a specific set of access points and may not generalize well to other environments or access point configurations. It is recommended to retrain the model if used in different locations."),
    ("03", "Result Interpretation", "The prediction is an estimate based on the learned Wi-Fi signal patterns. Accuracy may vary depending on environmental conditions and signal strength."),
    ("04", "Device Testing", "This prototype has only been tested on specific devices. Compatibility and potential issues with other devices cannot be guaranteed without comprehensive testing."),
]
notes_html = "".join(
    f"<article class='ips-note'><span class='mono n'>{n}</span><h3>{t}</h3><p>{p}</p></article>"
    for n, t, p in notes
)
st.markdown(
    compact(f"""
    <div class="ips"><section class="ips-notes">
      <p class="eyebrow">RESEARCH NOTES</p>
      <div class="ips-notes-grid">{notes_html}</div>
    </section></div>
    """),
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Run the scan (at the end so the whole page, with the locked button, renders first)
# --------------------------------------------------------------------------
if scanning:
    notices = []
    try:
        networks = get_campus_networks(SCAN_DURATION)
        frame = (
            pd.DataFrame(networks)
            .rename(columns={"ap_location": "Access Point", "rssi": "RSSI (dBm)"})
            .sort_values("RSSI (dBm)", ascending=False)
            .reset_index(drop=True)
        )
        st.session_state["scan"] = frame
        st.session_state["scan_time"] = datetime.now().strftime("%H:%M:%S")
        st.session_state["prediction"] = None

        if model is not None and scaler is not None:
            try:
                st.session_state["prediction"] = predict_location(networks, model, scaler)
            except Exception as exc:
                notices.append(("error", f"Prediction failed: {exc}"))
    except Exception as exc:
        notices.append(("error", f"Scan failed: {exc}"))
    finally:
        st.session_state["scanning"] = False
    st.session_state["scan_notices"] = notices
    st.rerun()