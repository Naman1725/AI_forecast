from flask import Flask, request, jsonify
import pandas as pd
import requests
from io import BytesIO
from prophet import Prophet
from datetime import timedelta

app = Flask(__name__)

# GitHub raw Excel URL
EXCEL_URL = "https://raw.githubusercontent.com/Naman1725/AI-forecast/main/data.xlsx"

def load_data():
    response = requests.get(EXCEL_URL)
    response.raise_for_status()
    return pd.read_excel(BytesIO(response.content))

@app.route("/forecast", methods=["GET"])
def forecast():
    try:
        # Get parameters
        country = request.args.get("country")
        technology = request.args.get("technology")
        zone = request.args.get("zone")
        kpi = request.args.get("kpi")
        forecast_months = int(request.args.get("forecast_time", 3))

        # Load and filter data
        df = load_data()
        df_filtered = df[
            (df["Country"] == country) &
            (df["Technology"] == technology) &
            (df["Zone"] == zone) &
            (df["KPI"] == kpi)
        ][["Month", "Value"]]

        if df_filtered.empty:
            return jsonify({"error": "No data found for given parameters"}), 404

        # Prepare for Prophet
        df_filtered = df_filtered.rename(columns={"Month": "ds", "Value": "y"})
        df_filtered["ds"] = pd.to_datetime(df_filtered["ds"])

        # Fit model
        model = Prophet()
        model.fit(df_filtered)

        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_months, freq="M")
        forecast_df = model.predict(future)

        # Split actual and forecasted
        last_actual_date = df_filtered["ds"].max()
        actual_data = forecast_df[forecast_df["ds"] <= last_actual_date][["ds", "yhat"]]
        forecast_data = forecast_df[forecast_df["ds"] > last_actual_date][["ds", "yhat"]]

        # Prepare Plotly-friendly JSON
        output = {
            "actual": {
                "x": actual_data["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "y": actual_data["yhat"].round(2).tolist(),
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
