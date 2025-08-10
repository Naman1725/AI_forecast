from flask import Flask, request, jsonify
import pandas as pd
import requests
from io import BytesIO
import zipfile
import joblib
import os
import re
import shutil

app = Flask(__name__)

# Corrected GitHub URLs
EXCEL_URL = "https://raw.githubusercontent.com/Naman1725/AI_forecast/main/data.xlsx"
MODELS_ZIP_URL = "https://raw.githubusercontent.com/Naman1725/AI_forecast/main/kpl_models.zip"  # Updated
MODELS_ZIP_PATH = "kpl_models.zip"  # Matches actual filename
MODELS_DIR = "kpi_models"

def safe_filename(s):
    """Create filesystem-safe filenames"""
    return re.sub(r'[^a-zA-Z0-9_]', '', s.replace(" ", "_"))

def load_excel():
    response = requests.get(EXCEL_URL)
    response.raise_for_status()
    return pd.read_excel(BytesIO(response.content))

def ensure_models_unzipped():
    """Download and extract models if needed"""
    if not os.path.exists(MODELS_DIR):
        # Download models zip
        r = requests.get(MODELS_ZIP_URL)
        r.raise_for_status()
        
        with open(MODELS_ZIP_PATH, "wb") as f:
            f.write(r.content)
        
        # Extract and rename directory
        with zipfile.ZipFile(MODELS_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall("temp_models")
        
        # Fix directory name mismatch
        os.rename("temp_models/models", MODELS_DIR)  # GitHub uses "models" directory
        shutil.rmtree("temp_models")

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        # Get parameters
        country = request.form.get("country", "Benin")
        technology = request.form.get("technology", "2G")
        zone = request.form.get("zone", "Zone 1")  # Note: Keep space in "Zone 1"
        kpi = request.form.get("kpi", "CSSR (Call Setup Success Rate)")
        forecast_months = int(request.form.get("forecast_time", 3))

        # Load historical data
        df = load_excel()
        
        # Convert to long format (your existing function)
        hist_df = convert_to_long_format(df, country, technology, zone, kpi)
        if hist_df.empty:
            return jsonify({"error": "No historical data found"}), 404
        
        # Ensure models are available
        ensure_models_unzipped()

        # Generate SAFE model filename
        model_name = f"{country} {zone}_{technology}_{kpi}".replace(" ", "_")
        model_name = safe_filename(model_name) + ".pkl"
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if not os.path.exists(model_path):
            return jsonify({
                "error": f"Model not found: {model_name}",
                "available_models": os.listdir(MODELS_DIR)
            }), 404

        # Load model and forecast
        model = joblib.load(model_path)
        future = model.make_future_dataframe(periods=forecast_months, freq='MS')
        forecast_df = model.predict(future)
        
        # Format response
        last_actual = hist_df["Month"].max()
        actuals = hist_df[hist_df["Month"] <= last_actual]
        forecast = forecast_df[forecast_df["ds"] > last_actual]
        
        return jsonify({
            "actual": {
                "x": actuals["Month"].dt.strftime("%Y-%m-%d").tolist(),
                "y": actuals["Value"].tolist()
            },
            "forecast": {
                "x": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "y": forecast["yhat"].tolist()
            }
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
