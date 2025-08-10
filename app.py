from flask import Flask, request, jsonify
import pandas as pd
import requests
from io import BytesIO
import zipfile
import joblib
from prophet import Prophet
import os
import re

app = Flask(__name__)

# URLs for GitHub raw files
EXCEL_URL = "https://raw.githubusercontent.com/Naman1725/AI_forecast/main/data.xlsx"
MODELS_ZIP_URL = "https://raw.githubusercontent.com/Naman1725/AI_forecast/main/models"

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
        r = requests.get(MODELS_ZIP_URL)
        r.raise_for_status()
        with open(MODELS_ZIP_PATH, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(MODELS_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODELS_DIR)

def convert_to_long_format(df, country, technology, zone, kpi):
    """Convert wide format data to long format with Month and Value columns."""
    # Filter the specific row
    filtered = df[
        (df["Country"].str.strip().str.lower() == country.strip().lower()) &
        (df["Technology"].str.strip().str.lower() == technology.strip().lower()) &
        (df["Zone"].str.strip().str.lower() == zone.strip().lower()) &
        (df["KPI"].str.strip().str.lower() == kpi.strip().lower())
    ]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Get month columns (all columns after 'Threshold')
    month_columns = [col for col in filtered.columns if re.match(r"[A-Za-z]{3}-\d{4}", str(col))]
    
    # Melt to long format
    long_df = filtered.melt(
        id_vars=["Country", "Technology", "Zone", "KPI"],
        value_vars=month_columns,
        var_name="Month",
        value_name="Value"
    )
    
    # Convert month strings to datetime
    long_df["Month"] = pd.to_datetime(long_df["Month"], format="%b-%Y")
    
    return long_df[["Month", "Value"]].sort_values("Month")

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        # Get parameters from form-data
        country = request.form.get("country")
        technology = request.form.get("technology")
        zone = request.form.get("zone")
        kpi = request.form.get("kpi")
        forecast_months = int(request.form.get("forecast_time", 3))

        if not all([country, technology, zone, kpi]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Load data
        df = load_excel()
        
        # Convert to long format
        hist_df = convert_to_long_format(df, country, technology, zone, kpi)
        
        if hist_df.empty:
            return jsonify({"error": "No data found for given parameters"}), 404

        # Ensure models extracted
        ensure_models_unzipped()

        # Build model filename
        safe_kpi = kpi.replace(" ", "_").replace("/", "_").replace("—", "_").replace("(", "").replace(")", "")
        model_filename = f"{country}_{zone}_{technology}_{safe_kpi}.pkl".replace(" ", "_")
        model_path = os.path.join(MODELS_DIR, model_filename)

        if not os.path.exists(model_path):
            return jsonify({
                "error": f"Model file not found: {model_filename}",
                "available_models": os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
            }), 404

        # Load model
        model = joblib.load(model_path)

        # Forecast
        future = model.make_future_dataframe(periods=forecast_months, freq='MS')
        forecast_df = model.predict(future)

        # Prepare output
        last_actual_date = hist_df["Month"].max()
        actual_data = hist_df[hist_df["Month"] <= last_actual_date]
        forecast_data = forecast_df[forecast_df["ds"] > last_actual_date][["ds", "yhat"]]

        return jsonify({
            "actual": {
                "x": actual_data["Month"].dt.strftime("%Y-%m-%d").tolist(),
                "y": actual_data["Value"].tolist(),
                "name": "Actual"
            },
            "forecast": {
                "x": forecast_data["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "y": forecast_data["yhat"].tolist(),
                "name": "Forecast"
            }
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
