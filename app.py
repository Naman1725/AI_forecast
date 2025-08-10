from flask import Flask, request, jsonify
import pandas as pd
import requests
from io import BytesIO
import zipfile
import joblib
from prophet import Prophet
import os

app = Flask(__name__)

# URLs for GitHub raw files
EXCEL_URL = "https://raw.githubusercontent.com/Naman1725/AI_forecast/main/data.xlsx"
MODELS_ZIP_URL = "https://raw.githubusercontent.com/Naman1725/AI_forecast/main/kpi_models.zip"

# Temp paths
MODELS_ZIP_PATH = "kpi_models.zip"
MODELS_DIR = "kpi_models"

def load_excel():
    """Load historical KPI data from GitHub."""
    response = requests.get(EXCEL_URL)
    response.raise_for_status()
    return pd.read_excel(BytesIO(response.content))

def ensure_models_unzipped():
    """Download and unzip models if not already extracted."""
    if not os.path.exists(MODELS_DIR):
        # Download
        r = requests.get(MODELS_ZIP_URL)
        r.raise_for_status()
        with open(MODELS_ZIP_PATH, "wb") as f:
            f.write(r.content)
        # Extract
        with zipfile.ZipFile(MODELS_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODELS_DIR)

@app.route("/forecast", methods=["GET"])
def forecast():
    try:
        # Get parameters
        country = request.args.get("country")
        technology = request.args.get("technology")
        zone = request.args.get("zone")
        kpi = request.args.get("kpi")
        forecast_months = int(request.args.get("forecast_time", 3))

        if not all([country, technology, zone, kpi]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Load data
        df = load_excel()

        # Filter historical data
        hist_df = df[
            (df["Country"] == country) &
            (df["Technology"] == technology) &
            (df["Zone"] == zone) &
            (df["KPI"] == kpi)
        ][["Month", "Value"]]

        if hist_df.empty:
            return jsonify({"error": "No data found for given parameters"}), 404

        # Prepare historical data
        hist_df["Month"] = pd.to_datetime(hist_df["Month"])
        hist_df = hist_df.sort_values("Month")

        # Ensure models extracted
        ensure_models_unzipped()

        # Build model filename
        safe_kpi = kpi.replace(" ", "_").replace("/", "_").replace("—", "_").replace("(", "").replace(")", "")
        model_filename = f"{country}_{zone}_{technology}_{safe_kpi}.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)

        if not os.path.exists(model_path):
            return jsonify({"error": f"Model file not found: {model_filename}"}), 404

        # Load model
        model = joblib.load(model_path)

        # Forecast
        future = model.make_future_dataframe(periods=forecast_months, freq='MS')
        forecast_df = model.predict(future)

        # Identify cutoff for actuals (Aug 2025)
        last_actual_date = hist_df["Month"].max()
        actual_data = hist_df[(hist_df["Month"] >= "2020-01-01") & (hist_df["Month"] <= last_actual_date)]

        forecast_data = forecast_df[forecast_df["ds"] > last_actual_date][["ds", "yhat"]]

        # Prepare output JSON
        output = {
            "actual": {
                "x": actual_data["Month"].dt.strftime("%Y-%m-%d").tolist(),
                "y": actual_data["Value"].round(2).tolist(),
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Actual"
            },
            "forecast": {
                "x": forecast_data["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "y": forecast_data["yhat"].round(2).tolist(),
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Forecast"
            }
        }

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
