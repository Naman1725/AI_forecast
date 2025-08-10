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
        # Immediately normalize column types to strings for consistent handling
        df.columns = [str(c) for c in df.columns]
        logger.debug("Excel loaded, shape=%s", df.shape)
        logger.debug("Excel columns (first 200): %s", df.columns.tolist()[:200])
        return df
    except Exception as e:
        logger.exception("Failed to download/load excel")
        raise RuntimeError(f"Failed to download/load excel: {e}")

def ensure_models_unzipped():
    """
    Use the MODELS_DIR (now set to 'models') if present.
    Only attempt download/extract if no local directory is found.
    """
    global MODELS_DIR

    # Look for common local model directories first
    candidates = [MODELS_DIR, "kpi_models", "kpl_models"]
    for cand in candidates:
        if cand and os.path.exists(cand) and os.listdir(cand):
            if cand != MODELS_DIR:
                logger.info("Found existing models directory '%s'. Using it instead of '%s'.", cand, MODELS_DIR)
                MODELS_DIR = cand
            else:
                logger.debug("Using models directory: %s", MODELS_DIR)
            return

    # No local folder found -> attempt download (fallback)
    logger.info("No local models folder found. Attempting to download models zip from %s", MODELS_ZIP_URL)
    try:
        r = requests.get(MODELS_ZIP_URL, timeout=60)
        r.raise_for_status()
    except Exception as e:
        logger.exception("Failed to download models zip")
        # re-check local dirs once more before bailing
        for cand in candidates:
            if cand and os.path.exists(cand) and os.listdir(cand):
                logger.info("After failed download, found local models dir '%s'. Using it.", cand)
                MODELS_DIR = cand
                return
        raise RuntimeError(f"Failed to download models zip: {e}")

    # Write zip and extract
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
        logger.debug("Extracted entries from zip: %s", entries)
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
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'[^0-9a-z]', '', s)
    return s

def detect_month_columns(columns):
    """Return list of column names that look like monthly columns.

    Heuristic: contains 4-digit year (e.g., 2020) OR starts with month name (Jan/Feb...)"""
    month_cols = []
    month_name_regex = re.compile(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', re.I)
    year_regex = re.compile(r'20\d{2}')
    for c in columns:
        cn = str(c)
        if year_regex.search(cn) or month_name_regex.search(cn) or re.search(r'\b\d{1,2}[-/]\d{4}\b', cn):
            month_cols.append(c)
    return month_cols

# -----------------------
# convert_to_long_format (with duplicate column name handling)
# -----------------------
def convert_to_long_format(df: pd.DataFrame, country: str, technology: str, zone: str, kpi: str) -> pd.DataFrame:
    """
    Convert a variety of Excel layouts into a canonical long DataFrame with columns:
      Month (datetime) and Value (numeric)
    Filtering is applied for country/technology/zone/kpi where columns exist.
    """
    df = df.copy()
    logger.debug("convert_to_long_format called with df.shape=%s", df.shape)

    # Normalize and dedupe column names
    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)
    if any(pd.Index(cols).duplicated()):
        logger.warning("Duplicate column labels detected before dedup: %s", cols)
        seen = {}
        new_cols = []
        for c in cols:
            name = str(c)
            if name in seen:
                seen[name] += 1
                new_name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
                new_name = name
            new_cols.append(new_name)
        df.columns = new_cols
        logger.info("Renamed duplicate columns. New columns (first 200): %s", df.columns.tolist()[:200])

    # If already in long format with Month + Value, try to filter and return
    if {"Month", "Value"}.issubset(set(df.columns)):
        logger.debug("Input already long format. Attempting to filter by metadata if present.")
        # Try to detect metadata columns with common names
        meta_cols = {c.lower(): c for c in df.columns}
        def pick_meta(names):
            for n in names:
                if n.lower() in meta_cols:
                    return meta_cols[n.lower()]
            # fallback: find any column containing the token
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
        # Apply filters if the columns exist
        if country_col:
            result = result[result[country_col].astype(str).apply(_normalize_str) == _normalize_str(country)]
        if zone_col:
            result = result[result[zone_col].astype(str).apply(_normalize_str) == _normalize_str(zone)]
        if tech_col:
            result = result[result[tech_col].astype(str).apply(_normalize_str) == _normalize_str(technology)]
        if kpi_col:
            result = result[result[kpi_col].astype(str).apply(_normalize_str) == _normalize_str(kpi)]

        # Convert and return
        if "Month" in result.columns and "Value" in result.columns:
            result["Month"] = pd.to_datetime(result["Month"], errors="coerce")
            result["Value"] = pd.to_numeric(result["Value"], errors="coerce")
            result = result.dropna(subset=["Month", "Value"]).sort_values("Month")
            logger.debug("Returning long-format result shape=%s", result.shape)
            return result[["Month", "Value"]]

    # Detect month-like columns
    month_cols = detect_month_columns(df.columns)
    logger.debug("Detected month columns: %s", month_cols)

    if not month_cols:
        # fallback: columns containing any 4-digit year
        candidate = [c for c in df.columns if re.search(r'\b20\d{2}\b', str(c))]
        if candidate:
            month_cols = candidate
            logger.debug("Fallback month columns via 4-digit year match: %s", month_cols)

    if not month_cols:
        raise ValueError(
            "Could not detect monthly columns in the Excel file. "
            "Column names found: " + ", ".join([str(c) for c in df.columns[:200]])
        )

    id_vars = [c for c in df.columns if c not in month_cols]
    if len(id_vars) == 0:
        # create dummy id column
        df["_row_id"] = range(len(df))
        id_vars = ["_row_id"]

    # Melt wide -> long
    melted = df.melt(id_vars=id_vars, value_vars=month_cols, var_name="Month", value_name="Value")
    logger.debug("Melted shape=%s", melted.shape)

    # Attempt to locate metadata columns among id_vars
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

    logger.debug("Detected metadata columns => country:%s zone:%s tech:%s kpi:%s",
                 country_col, zone_col, tech_col, kpi_col)

    # Apply filters where columns exist
    if country_col:
        melted = melted[melted[country_col].astype(str).apply(_normalize_str) == _normalize_str(country)]
    if zone_col:
        melted = melted[melted[zone_col].astype(str).apply(_normalize_str) == _normalize_str(zone)]
    if tech_col:
        melted = melted[melted[tech_col].astype(str).apply(_normalize_str) == _normalize_str(technology)]
    if kpi_col:
        melted = melted[melted[kpi_col].astype(str).apply(_normalize_str) == _normalize_str(kpi)]

    # Convert Month to datetime into a dedicated column, but DO NOT overwrite the original 'Month' string column
    melted["Month_parsed"] = pd.to_datetime(melted["Month"], errors="coerce", dayfirst=False)
    if melted["Month_parsed"].isna().all():
        # try some common formats
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

    # IMPORTANT: build the final result from Month_parsed only so we never end up with duplicate 'Month' labels
    result = melted.loc[:, ["Month_parsed", "Value"]].rename(columns={"Month_parsed": "Month"})
    result = result.dropna(subset=["Month", "Value"]).sort_values("Month").reset_index(drop=True)

    logger.debug("convert_to_long_format returning shape=%s", result.shape)
    return result

def validate_hist_df(hist_df):
    """Ensure hist_df contains expected columns Month (datetime) and Value (numeric)"""
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
    """List files in the models directory for debugging"""
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
        # Parse params
        country = request.form.get("country", "Benin")
        technology = request.form.get("technology", "2G")
        zone = request.form.get("zone", "Zone 1")
        kpi = request.form.get("kpi", "CSSR (Call Setup Success Rate)")
        try:
            forecast_months = int(request.form.get("forecast_time", 3))
        except Exception:
            return jsonify({"error": "forecast_time must be integer"}), 400

        # Load historical data
        df = load_excel()

        # Convert to long format
        try:
            hist_df = convert_to_long_format(df, country, technology, zone, kpi)
        except Exception as e:
            logger.exception("convert_to_long_format failed")
            payload = {
                "error": "convert_to_long_format failed",
                "message": str(e),
                "hint": "Ensure your Excel has monthly columns (e.g., 'Jan-2020' or '2020-01') or a long table with 'Month' and 'Value'.",
                "detected_columns": df.columns.tolist()[:200]
            }
            if DEBUG:
                payload["traceback"] = traceback.format_exc()
            return jsonify(payload), 400

        # Validate
        hist_df = validate_hist_df(hist_df)
        if hist_df.empty:
            return jsonify({"error": "No historical data found for the given parameters"}), 404

        # Ensure models available (will use MODELS_DIR = "models" you uploaded)
        ensure_models_unzipped()

        # Build model filename and load
        
        # Instead of replacing all spaces blindly, build manually
        zone_str = zone  # keep the space between Zone and number intact
        model_name = f"{country}_{zone_str}_{technology}_{kpi}"
        model_name = safe_filename(model_name) + ".pkl"

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
            payload = {"error": f"Failed to load model: {e}"}
            if DEBUG:
                payload["traceback"] = traceback.format_exc()
            return jsonify(payload), 500

        # Forecast - expecting Prophet-like interface
        try:
            future = model.make_future_dataframe(periods=forecast_months, freq='MS')
            forecast_df = model.predict(future)
        except AttributeError:
            logger.exception("Model does not support Prophet API (make_future_dataframe/predict).")
            return jsonify({"error": "Model does not support Prophet API (make_future_dataframe/predict)."}), 500
        except Exception as e:
            logger.exception("Prediction failed")
            payload = {"error": f"Prediction failed: {e}"}
            if DEBUG:
                payload["traceback"] = traceback.format_exc()
            return jsonify(payload), 500

        # Prepare response
        try:
            last_actual = hist_df["Month"].max()
            actuals = hist_df[hist_df["Month"] <= last_actual].sort_values("Month")
            forecast_rows = forecast_df[forecast_df["ds"] > last_actual].sort_values("ds")
        except Exception as e:
            logger.exception("Failed preparing response slices")
            payload = {"error": f"Failed preparing response: {e}"}
            if DEBUG:
                payload["traceback"] = traceback.format_exc()
            return jsonify(payload), 500

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
    # For local testing only. On Render the WSGI server is used.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=DEBUG)
