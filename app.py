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
MODELS_DIR = "kpi_models"

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
    return re.sub(r'[^a-zA-Z0-9_]', '', s.replace(" ", "_"))

def load_excel():
    try:
        logger.debug("Downloading Excel from: %s", EXCEL_URL)
        resp = requests.get(EXCEL_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(BytesIO(resp.content))
        logger.debug("Excel loaded, shape=%s", df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to download/load excel")
        raise RuntimeError(f"Failed to download/load excel: {e}")

def ensure_models_unzipped():
    if os.path.exists(MODELS_DIR) and os.listdir(MODELS_DIR):
        logger.debug("Models directory already exists: %s", MODELS_DIR)
        return

    logger.info("Models directory missing or empty. Downloading models zip...")
    try:
        r = requests.get(MODELS_ZIP_URL, timeout=60)
        r.raise_for_status()
    except Exception as e:
        logger.exception("Failed to download models zip")
        raise RuntimeError(f"Failed to download models zip: {e}")

    with open(MODELS_ZIP_PATH, "wb") as f:
        f.write(r.content)
    logger.debug("Models zip written to %s", MODELS_ZIP_PATH)

    temp_dir = "temp_models"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(MODELS_ZIP_PATH, "r") as z:
            z.extractall(temp_dir)
        entries = os.listdir(temp_dir)
        logger.debug("Extracted entries: %s", entries)
        candidate = None
        for e in entries:
            full = os.path.join(temp_dir, e)
            if os.path.isdir(full):
                if e.lower() == "models":
                    candidate = full
                    break
                candidate = full if candidate is None else candidate
        src = candidate or temp_dir
        if os.path.exists(MODELS_DIR):
            shutil.rmtree(MODELS_DIR)
        shutil.move(src, MODELS_DIR)
        logger.info("Models moved to %s", MODELS_DIR)
    except Exception as e:
        logger.exception("Failed to extract/move models")
        raise RuntimeError(f"Failed to extract/move models: {e}")
    finally:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(MODELS_ZIP_PATH):
                os.remove(MODELS_ZIP_PATH)
        except Exception:
            pass

def _normalize_str(x: Optional[object]) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.strip().lower()
    s = re.sub(r'\s+', '', s)  # remove spaces
    s = re.sub(r'[^0-9a-z]', '', s)
    return s

def detect_month_columns(columns):
    """Return list of column names that look like monthly columns.
    Heuristic: column contains 4-digit year (e.g., 2020) OR month names."""
    month_cols = []
    month_name_regex = re.compile(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', re.I)
    year_regex = re.compile(r'20\d{2}')
    for c in columns:
        cn = str(c)
        if year_regex.search(cn) or month_name_regex.search(cn):
            month_cols.append(c)
    return month_cols

# -----------------------
# convert_to_long_format (inlined to avoid missing-import error)
# -----------------------
def convert_to_long_format(df: pd.DataFrame, country: str, technology: str, zone: str, kpi: str) -> pd.DataFrame:
    """
    Convert a variety of Excel layouts into a canonical long DataFrame with columns:
      Month (datetime) and Value (numeric)
    Filtering is applied for country/technology/zone/kpi where columns exist.
    """
    df = df.copy()
    logger.debug("convert_to_long_format called with df.shape=%s", df.shape)

    # If already in long format, require Month + Value
    if {"Month", "Value"}.issubset(set(df.columns)):
        logger.debug("Input already long format. Attempting to filter.")
        # Attempt to filter on other available metadata columns if present
        meta_filters = {}
        for name, val in [("Country", country), ("country", country),
                          ("Zone", zone), ("zone", zone),
                          ("Technology", technology), ("technology", technology),
                          ("KPI", kpi), ("kpi", kpi)]:
            if name in df.columns:
                meta_filters[name] = val

        for col, val in meta_filters.items():
            if val is None or val == "":
                continue
            df = df[df[col].astype(str).apply(_normalize_str) == _normalize_str(val)]
        result = df[["Month", "Value"]].copy()
        result["Month"] = pd.to_datetime(result["Month"], errors="coerce")
        result["Value"] = pd.to_numeric(result["Value"], errors="coerce")
        result = result.dropna(subset=["Month", "Value"]).sort_values("Month")
        logger.debug("Returning long-format result shape=%s", result.shape)
        return result

    # If columns are a MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(i) for i in col if i and str(i).strip() != ""]) for col in df.columns.values]
        logger.debug("Flattened MultiIndex columns.")

    # Detect month-like columns heuristically
    month_cols = detect_month_columns(df.columns)
    logger.debug("Detected month columns: %s", month_cols)

    if not month_cols:
        # last attempt: detect columns that look like YYYY-MM or MMM-YYYY with common separators
        candidate = [c for c in df.columns if re.search(r'\d{4}', str(c))]
        if candidate:
            month_cols = candidate
            logger.debug("Fallback month columns via any 4-digit match: %s", month_cols)

    if not month_cols:
        raise ValueError(
            "Could not detect monthly columns in the Excel file. "
            "Column names found: " + ", ".join([str(c) for c in df.columns[:50]])
        )

    id_vars = [c for c in df.columns if c not in month_cols]
    if len(id_vars) == 0:
        # If there are no id_vars, create a dummy id column so melt works
        df["_row_id"] = range(len(df))
        id_vars = ["_row_id"]

    # Melt wide -> long
    melted = df.melt(id_vars=id_vars, value_vars=month_cols, var_name="Month", value_name="Value")
    logger.debug("Melted shape=%s", melted.shape)

    # Normalize column names to find metadata columns (country/zone/etc.)
    col_map = {c: c for c in id_vars}
    lower_to_orig = {c.lower(): c for c in id_vars}

    # find best candidates
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

    # Apply filters where columns exist
    filters = []
    if country_col:
        filters.append((country_col, country))
    if zone_col:
        filters.append((zone_col, zone))
    if tech_col:
        filters.append((tech_col, technology))
    if kpi_col:
        filters.append((kpi_col, kpi))

    logger.debug("Detected metadata columns => country:%s zone:%s tech:%s kpi:%s",
                 country_col, zone_col, tech_col, kpi_col)

    for col, val in filters:
        if val is None or str(val).strip() == "":
            continue
        melted = melted[melted[col].astype(str).apply(_normalize_str) == _normalize_str(val)]

    # Convert Month to datetime
    melted["Month_parsed"] = pd.to_datetime(melted["Month"], errors="coerce", dayfirst=False)
    # If all NaT, try some alternative parsing (month name + year)
    if melted["Month_parsed"].isna().all():
        # try common formats
        def try_parse(s):
            s = str(s)
            for fmt in ("%b-%Y", "%B-%Y", "%Y-%m", "%Y/%m", "%m/%Y", "%Y"):
                try:
                    return pd.to_datetime(s, format=fmt)
                except Exception:
                    continue
            try:
                return pd.to_datetime(s, errors="coerce")
            except Exception:
                return pd.NaT
        melted["Month_parsed"] = melted["Month"].apply(try_parse)

    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
    result = melted.dropna(subset=["Month_parsed", "Value"]).rename(columns={"Month_parsed": "Month"})
    result = result[["Month", "Value"]].sort_values("Month").reset_index(drop=True)
    logger.debug("convert_to_long_format returning shape=%s", result.shape)
    return result

def validate_hist_df(hist_df):
    if hist_df is None or len(hist_df) == 0:
        raise ValueError("Historical dataframe is empty.")
    if "Month" not in hist_df.columns or "Value" not in hist_df.columns:
        raise ValueError("Historical dataframe must contain 'Month' and 'Value' columns.")
    if not pd.api.types.is_datetime64_any_dtype(hist_df["Month"]):
        try:
            hist_df["Month"] = pd.to_datetime(hist_df["Month"])
        except Exception:
            raise ValueError("'Month' column could not be converted to datetime.")
    if not pd.api.types.is_numeric_dtype(hist_df["Value"]):
        try:
            hist_df["Value"] = pd.to_numeric(hist_df["Value"])
        except Exception:
            raise ValueError("'Value' column could not be converted to numeric.")
    return hist_df

# -----------------------
# Routes
# -----------------------
@app.route("/models", methods=["GET"])
def list_models():
    try:
        if not os.path.exists(MODELS_DIR):
            return jsonify({"models_dir_exists": False, "message": f"{MODELS_DIR} not found on disk"}), 200
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

        # Convert to long format (now inlined above)
        try:
            hist_df = convert_to_long_format(df, country, technology, zone, kpi)
        except Exception as e:
            logger.exception("convert_to_long_format failed")
            # return helpful error to caller
            return jsonify({
                "error": "convert_to_long_format failed",
                "message": str(e),
                "hint": "Ensure your Excel has monthly columns (e.g., 'Jan-2020' or '2020-01') or a long table with 'Month' and 'Value'."
            }), 400

        hist_df = validate_hist_df(hist_df)

        if hist_df.empty:
            return jsonify({"error": "No historical data found for the given parameters"}), 404

        ensure_models_unzipped()

        model_name = f"{country} {zone}_{technology}_{kpi}".replace(" ", "_")
        model_name = safe_filename(model_name) + ".pkl"
        model_path = os.path.join(MODELS_DIR, model_name)
        logger.debug("Looking for model at: %s", model_path)

        if not os.path.exists(model_path):
            available = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
            logger.error("Model not found: %s", model_name)
            return jsonify({
                "error": f"Model not found: {model_name}",
                "available_models": available
            }), 404

        try:
            model = joblib.load(model_path)
            logger.info("Model loaded: %s", model_path)
        except Exception as e:
            logger.exception("Failed to load model")
            return jsonify({"error": f"Failed to load model: {e}"}), 500

        try:
            future = model.make_future_dataframe(periods=forecast_months, freq='MS')
            forecast_df = model.predict(future)
        except AttributeError:
            logger.exception("Model does not support Prophet API (make_future_dataframe/predict).")
            return jsonify({"error": "Model does not support Prophet API (make_future_dataframe/predict)."}), 500
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({"error": f"Prediction failed: {e}"}), 500

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
                "y": forecast_rows["yhat"].tolist() if "yhat" in forecast_rows.columns else forecast_rows.iloc[:, 1].tolist()
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
