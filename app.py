# app.py
import os
import re
import shutil
import zipfile
import traceback
import logging
from io import BytesIO
from typing import Optional

import requests
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# -----------------------
# Config
# -----------------------
EXCEL_URL = "https://raw.githubusercontent.com/Naman1725/AI_forecast/main/data.xlsx"
MODELS_ZIP_URL = "https://raw.githubusercontent.com/Naman1725/AI_forecast/main/kpl_models.zip"
MODELS_ZIP_PATH = "kpl_models.zip"
MODELS_DIR = "models"  # <-- changed to use your uploaded unzipped models folder

DEBUG = os.environ.get("DEBUG", "1") == "1"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# -----------------------
# Helpers
# -----------------------
def safe_filename(s: str) -> str:
    """Create filesystem-safe filenames, preserving spaces in zone names until join step"""
    return re.sub(r'[^a-zA-Z0-9_ ]', '', s)  # allow spaces temporarily


def load_excel():
    """Download the excel file and return a pandas DataFrame"""
    try:
        logger.debug("Downloading Excel from: %s", EXCEL_URL)
        resp = requests.get(EXCEL_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(BytesIO(resp.content))
        df.columns = [str(c) for c in df.columns]  # normalize column types
        logger.debug("Excel loaded, shape=%s", df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to download/load excel")
        raise RuntimeError(f"Failed to download/load excel: {e}")

def ensure_models_unzipped():
    """Ensure models are available locally, download/unzip if needed"""
    global MODELS_DIR
    candidates = [MODELS_DIR, "kpi_models", "kpl_models"]
    for cand in candidates:
        if cand and os.path.exists(cand) and os.listdir(cand):
            if cand != MODELS_DIR:
                logger.info("Found existing models directory '%s'. Using it.", cand)
                MODELS_DIR = cand
            return

    logger.info("No local models folder found. Attempting to download models zip from %s", MODELS_ZIP_URL)
    try:
        r = requests.get(MODELS_ZIP_URL, timeout=60)
        r.raise_for_status()
    except Exception as e:
        logger.exception("Failed to download models zip")
        for cand in candidates:
            if cand and os.path.exists(cand) and os.listdir(cand):
                logger.info("After failed download, found local models dir '%s'. Using it.", cand)
                MODELS_DIR = cand
                return
        raise RuntimeError(f"Failed to download models zip: {e}")

    with open(MODELS_ZIP_PATH, "wb") as f:
        f.write(r.content)

    temp_dir = "temp_models"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(MODELS_ZIP_PATH, "r") as z:
            z.extractall(temp_dir)
        entries = os.listdir(temp_dir)
        candidate = None
        for e in entries:
            full = os.path.join(temp_dir, e)
            if os.path.isdir(full):
                if e.lower() == "models":
                    candidate = full
                    break
                if candidate is None:
                    candidate = full
        src = candidate or temp_dir
        if os.path.exists(MODELS_DIR):
            shutil.rmtree(MODELS_DIR)
        shutil.move(src, MODELS_DIR)
        logger.info("Models moved to %s", MODELS_DIR)
    except Exception as e:
        logger.exception("Failed to extract/move models")
        raise RuntimeError(f"Failed to extract/move models: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(MODELS_ZIP_PATH):
            os.remove(MODELS_ZIP_PATH)

def _normalize_str(x: Optional[object]) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'[^0-9a-z]', '', s)
    return s

def detect_month_columns(columns):
    month_cols = []
    month_name_regex = re.compile(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', re.I)
    year_regex = re.compile(r'20\d{2}')
    for c in columns:
        cn = str(c)
        if year_regex.search(cn) or month_name_regex.search(cn) or re.search(r'\b\d{1,2}[-/]\d{4}\b', cn):
            month_cols.append(c)
    return month_cols

def convert_to_long_format(df: pd.DataFrame, country: str, technology: str, zone: str, kpi: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)
    if any(pd.Index(cols).duplicated()):
        seen = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols

    if {"Month", "Value"}.issubset(set(df.columns)):
        meta_cols = {c.lower(): c for c in df.columns}
        def pick_meta(names):
            for n in names:
                if n.lower() in meta_cols:
                    return meta_cols[n.lower()]
            for c in df.columns:
                for token in names:
                    if token.lower() in c.lower():
                        return c
            return None

        country_col = pick_meta(["country", "countryname", "nation"])
        zone_col = pick_meta(["zone", "region", "state", "area"])
        tech_col = pick_meta(["technology", "tech", "technology_name"])
        kpi_col = pick_meta(["kpi", "metric", "measure"])

        result = df.copy()
        if country_col:
            result = result[result[country_col].astype(str).apply(_normalize_str) == _normalize_str(country)]
        if zone_col:
            result = result[result[zone_col].astype(str).apply(_normalize_str) == _normalize_str(zone)]
        if tech_col:
            result = result[result[tech_col].astype(str).apply(_normalize_str) == _normalize_str(technology)]
        if kpi_col:
            result = result[result[kpi_col].astype(str).apply(_normalize_str) == _normalize_str(kpi)]

        result["Month"] = pd.to_datetime(result["Month"], errors="coerce")
        result["Value"] = pd.to_numeric(result["Value"], errors="coerce")
        return result.dropna(subset=["Month", "Value"]).sort_values("Month")[["Month", "Value"]]

    month_cols = detect_month_columns(df.columns)
    if not month_cols:
        candidate = [c for c in df.columns if re.search(r'\b20\d{2}\b', str(c))]
        if candidate:
            month_cols = candidate
    if not month_cols:
        raise ValueError("Could not detect monthly columns.")

    id_vars = [c for c in df.columns if c not in month_cols]
    if not id_vars:
        df["_row_id"] = range(len(df))
        id_vars = ["_row_id"]

    melted = df.melt(id_vars=id_vars, value_vars=month_cols, var_name="Month", value_name="Value")

    def pick_column(possible_names):
        for n in possible_names:
            for c in id_vars:
                if n in str(c).lower():
                    return c
        return None

    country_col = pick_column(["country", "countryname", "nation"])
    zone_col = pick_column(["zone", "region", "state", "area"])
    tech_col = pick_column(["technology", "tech", "technology_name"])
    kpi_col = pick_column(["kpi", "metric", "measure"])

    if country_col:
        melted = melted[melted[country_col].astype(str).apply(_normalize_str) == _normalize_str(country)]
    if zone_col:
        melted = melted[melted[zone_col].astype(str).apply(_normalize_str) == _normalize_str(zone)]
    if tech_col:
        melted = melted[melted[tech_col].astype(str).apply(_normalize_str) == _normalize_str(technology)]
    if kpi_col:
        melted = melted[melted[kpi_col].astype(str).apply(_normalize_str) == _normalize_str(kpi)]

    melted["Month_parsed"] = pd.to_datetime(melted["Month"], errors="coerce")
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")

    result = melted[["Month_parsed", "Value"]].rename(columns={"Month_parsed": "Month"})
    return result.dropna(subset=["Month", "Value"]).sort_values("Month").reset_index(drop=True)

def validate_hist_df(hist_df):
    if hist_df is None or len(hist_df) == 0:
        raise ValueError("Historical dataframe is empty.")
    if "Month" not in hist_df.columns or "Value" not in hist_df.columns:
        raise ValueError("Historical dataframe must contain 'Month' and 'Value'.")
    if not pd.api.types.is_datetime64_any_dtype(hist_df["Month"]):
        hist_df["Month"] = pd.to_datetime(hist_df["Month"])
    if not pd.api.types.is_numeric_dtype(hist_df["Value"]):
        hist_df["Value"] = pd.to_numeric(hist_df["Value"])
    return hist_df

# -----------------------
# Routes
# -----------------------
@app.route("/models", methods=["GET"])
def list_models():
    try:
        if not os.path.exists(MODELS_DIR):
            return jsonify({"models_dir_exists": False, "message": f"{MODELS_DIR} not found"}), 200
        files = os.listdir(MODELS_DIR)
        return jsonify({"models_dir_exists": True, "models": files}), 200
    except Exception as e:
        logger.exception("Error listing models")
        payload = {"error": str(e)}
        if DEBUG:
            payload["traceback"] = traceback.format_exc()
        return jsonify(payload), 500

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        country = request.form.get("country", "Benin")
        technology = request.form.get("technology", "2G")
        zone = request.form.get("zone", "Zone 1")
        kpi = request.form.get("kpi", "CSSR (Call Setup Success Rate)")
        try:
            forecast_months = int(request.form.get("forecast_time", 3))
        except Exception:
            return jsonify({"error": "forecast_time must be integer"}), 400

        df = load_excel()
        try:
            hist_df = convert_to_long_format(df, country, technology, zone, kpi)
        except Exception as e:
            logger.exception("convert_to_long_format failed")
            payload = {"error": "convert_to_long_format failed", "message": str(e)}
            if DEBUG:
                payload["traceback"] = traceback.format_exc()
            return jsonify(payload), 400

        hist_df = validate_hist_df(hist_df)
        if hist_df.empty:
            return jsonify({"error": "No historical data found"}), 404

        ensure_models_unzipped()
        zone_str = zone
        model_name = safe_filename(f"{country}{zone_str}{technology}_{kpi}") + ".pkl"
        model_path = os.path.join(MODELS_DIR, model_name)
        logger.debug("Looking for model at: %s", model_path)

        if not os.path.exists(model_path):
            available = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
            return jsonify({"error": f"Model not found: {model_name}", "available_models": available}), 404

        try:
            model = joblib.load(model_path)
        except Exception as e:
            payload = {"error": f"Failed to load model: {e}"}
            if DEBUG:
                payload["traceback"] = traceback.format_exc()
            return jsonify(payload), 500

        try:
            future = model.make_future_dataframe(periods=forecast_months, freq='MS')
            forecast_df = model.predict(future)
        except Exception as e:
            payload = {"error": f"Prediction failed: {e}"}
            if DEBUG:
                payload["traceback"] = traceback.format_exc()
            return jsonify(payload), 500

        last_actual = hist_df["Month"].max()
        actuals = hist_df[hist_df["Month"] <= last_actual].sort_values("Month")
        forecast_rows = forecast_df[forecast_df["ds"] > last_actual].sort_values("ds")

        response = {
            "actual": {
                "x": actuals["Month"].dt.strftime("%Y-%m-%d").tolist(),
                "y": actuals["Value"].tolist()
            },
            "forecast": {
                "x": forecast_rows["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "y": forecast_rows["yhat"].tolist()
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Unhandled error in /forecast")
        payload = {"error_type": type(e).__name__, "error": str(e)}
        if DEBUG:
            payload["traceback"] = traceback.format_exc()
        return jsonify(payload), 500

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=DEBUG)
