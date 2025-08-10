# app.py
import os
import re
import shutil
import zipfile
import traceback
import logging
from io import BytesIO

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

# Debug flag controlled by env var DEBUG (set to "1" for verbose tracebacks)
DEBUG = os.environ.get("DEBUG", "1") == "1"

# Logging config
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
    """Create filesystem-safe filenames"""
    return re.sub(r'[^a-zA-Z0-9_]', '', s.replace(" ", "_"))

def load_excel():
    """Download the excel file and return a pandas DataFrame"""
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
    """Download and extract models if MODELS_DIR doesn't exist"""
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

    # Write zip
    with open(MODELS_ZIP_PATH, "wb") as f:
        f.write(r.content)
    logger.debug("Models zip written to %s", MODELS_ZIP_PATH)

    # Extract into a temp dir and locate the extracted directory
    temp_dir = "temp_models"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(MODELS_ZIP_PATH, "r") as z:
            z.extractall(temp_dir)
        # Find first directory inside temp_dir that looks like a models dir
        entries = os.listdir(temp_dir)
        logger.debug("Extracted entries: %s", entries)
        candidate = None
        for e in entries:
            full = os.path.join(temp_dir, e)
            if os.path.isdir(full):
                # If there's a child dir named 'models' use that first
                if e.lower() == "models":
                    candidate = full
                    break
                candidate = full if candidate is None else candidate

        # If zip contained files directly (no dir), just use temp_dir itself
        src = candidate or temp_dir

        if os.path.exists(MODELS_DIR):
            shutil.rmtree(MODELS_DIR)
        shutil.move(src, MODELS_DIR)
        logger.info("Models moved to %s", MODELS_DIR)
    except Exception as e:
        logger.exception("Failed to extract/move models")
        raise RuntimeError(f"Failed to extract/move models: {e}")
    finally:
        # cleanup
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(MODELS_ZIP_PATH):
                os.remove(MODELS_ZIP_PATH)
        except Exception:
            pass

# Try importing convert_to_long_format from a helper module if present
try:
    from data_utils import convert_to_long_format  # optional helper module
    logger.debug("Imported convert_to_long_format from data_utils")
except Exception:
    logger.debug("data_utils.convert_to_long_format not available; expecting it in this file's scope")

def validate_hist_df(hist_df):
    """Ensure hist_df contains expected columns Month (datetime) and Value (numeric)"""
    if hist_df is None or len(hist_df) == 0:
        raise ValueError("Historical dataframe is empty.")
    if "Month" not in hist_df.columns or "Value" not in hist_df.columns:
        raise ValueError("Historical dataframe must contain 'Month' and 'Value' columns.")
    # Ensure Month is datetime
    if not pd.api.types.is_datetime64_any_dtype(hist_df["Month"]):
        try:
            hist_df["Month"] = pd.to_datetime(hist_df["Month"])
        except Exception:
            raise ValueError("'Month' column could not be converted to datetime.")
    # Ensure Value is numeric
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

        # Ensure convert_to_long_format exists
        if "convert_to_long_format" not in globals():
            msg = ("convert_to_long_format function not found. "
                   "Either define it in this file or provide a data_utils.py with that function.")
            logger.error(msg)
            return jsonify({"error": msg}), 500

        # Convert to long format
        try:
            hist_df = convert_to_long_format(df, country, technology, zone, kpi)
        except Exception as e:
            logger.exception("convert_to_long_format raised an exception")
            raise RuntimeError(f"convert_to_long_format failed: {e}")

        # Validate hist_df layout
        hist_df = validate_hist_df(hist_df)

        if hist_df.empty:
            return jsonify({"error": "No historical data found for the given parameters"}), 404

        # Ensure models are present
        ensure_models_unzipped()

        # Build model filename
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

        # Load model
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded: %s", model_path)
        except Exception as e:
            logger.exception("Failed to load model")
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        # Make future dataframe and predict
        try:
            # This assumes the model is a Prophet-like object. If you've used a different interface,
            # adjust these calls accordingly.
            future = model.make_future_dataframe(periods=forecast_months, freq='MS')
            forecast_df = model.predict(future)
        except AttributeError:
            # If model doesn't have Prophet API, attempt alternative predict interface
            logger.exception("Model does not support Prophet API (make_future_dataframe/predict).")
            raise RuntimeError("Model does not support Prophet API (make_future_dataframe/predict).")
        except Exception as e:
            logger.exception("Failed during prediction")
            raise RuntimeError(f"Prediction failed: {e}")

        # Prepare response: align actuals and forecasts
        try:
            last_actual = hist_df["Month"].max()
            actuals = hist_df[hist_df["Month"] <= last_actual].sort_values("Month")
            # Prophet's forecast uses 'ds' for dates and 'yhat' for predicted values
            forecast_rows = forecast_df[forecast_df["ds"] > last_actual].sort_values("ds")
        except Exception as e:
            logger.exception("Failed preparing response slices")
            raise RuntimeError(f"Failed preparing response: {e}")

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
