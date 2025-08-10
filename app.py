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
    """Load historical KPI data from GitHub with robust column handling."""
    response = requests.get(EXCEL_URL)
    response.raise_for_status()
    
    # Load Excel and clean column names
    df = pd.read_excel(BytesIO(response.content))
    df.columns = [col.strip().lower() for col in df.columns]
    return df

def ensure_models_unzipped():
    """Download and unzip models if not already extracted."""
    if not os.path.exists(MODELS_DIR):
        r = requests.get(MODELS_ZIP_URL)
        r.raise_for_status()
        with open(MODELS_ZIP_PATH, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(MODELS_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODELS_DIR)

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        # Get parameters
        country = request.form.get("country")
        technology = request.form.get("technology")
        zone = request.form.get("zone")
        kpi = request.form.get("kpi")
        forecast_months = int(request.form.get("forecast_time", 3))

        if not all([country, technology, zone, kpi]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Load and clean data
        df = load_excel()
        
        # Check for required columns
        required_cols = ['country', 'technology', 'zone', 'kpi', 'month', 'value']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({
                "error": f"Missing columns in data: {missing}",
                "available_columns": list(df.columns)
            }), 500

        # Filter historical data
        hist_df = df[
            (df["country"].str.strip().str.lower() == country.strip().lower()) &
            (df["technology"].str.strip().str.lower() == technology.strip().lower()) &
            (df["zone"].str.strip().str.lower() == zone.strip().lower()) &
            (df["kpi"].str.strip().str.lower() == kpi.strip().lower())
        ][["month", "value"]]

        if hist_df.empty:
            return jsonify({"error": "No data found for given parameters"}), 404

        # Prepare historical data
        hist_df["month"] = pd.to_datetime(hist_df["month"])
        hist_df = hist_df.sort_values("month")

        # Ensure models extracted
        ensure_models_unzipped()

        # Build model filename
        safe_kpi = kpi.replace(" ", "_").replace("/", "_").replace("—", "_").replace("(", "").replace(")", "")
        model_filename = f"{country}_{zone}_{technology}_{safe_kpi}.pkl".lower()
        model_path = os.path.join(MODELS_DIR, model_filename)

        if not os.path.exists(model_path):
            return jsonify({
                "error": f"Model file not found: {model_filename}",
                "available_models": os.listdir(MODELS_DIR)
            }), 404

        # Load model
        model = joblib.load(model_path)

        # Forecast
        future = model.make_future_dataframe(periods=forecast_months, freq='MS')
        forecast_df = model.predict(future)

        # Prepare output
        last_actual_date = hist_df["month"].max()
        actual_data = hist_df[hist_df["month"] <= last_actual_date]
        forecast_data = forecast_df[forecast_df["ds"] > last_actual_date][["ds", "yhat"]]

        return jsonify({
            "actual": {
                "x": actual_data["month"].dt.strftime("%Y-%m-%d").tolist(),
                "y": actual_data["value"].round(2).tolist(),
                "name": "Actual"
            },
            "forecast": {
                "x": forecast_data["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "y": forecast_data["yhat"].round(2).tolist(),
                "name": "Forecast"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
