import os
import re
import joblib
import logging
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path where models are stored
MODELS_DIR = os.path.join(os.getcwd(), "models")

# Safe filename function
def safe_filename(s: str) -> str:
    """Remove any character not alphanumeric, underscore, or space."""
    return re.sub(r'[^a-zA-Z0-9_ ]', '', s)

# Relaxed filename normalization for tolerant matching
def norm_alnum(s: str) -> str:
    """Lowercase, strip non-alphanumeric."""
    return re.sub(r'[^0-9a-z]', '', s.lower())

def resolve_model_path(country, zone, technology, kpi):
    """Find the best matching model file using tolerant matching."""
    zone_str = f"Zone {zone}" if not str(zone).lower().startswith("zone") else zone
    raw_model_name = f"{country}_{zone_str}_{technology}_{kpi}"
    expected = safe_filename(raw_model_name).strip()
    expected_filename = expected + ".pkl"
    model_path = os.path.join(MODELS_DIR, expected_filename)

    if os.path.exists(model_path):
        return model_path, expected_filename, None  # exact match

    # Tolerant matching
    candidates = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
    target_norm = norm_alnum(expected)
    for f in candidates:
        base = f[:-4] if f.lower().endswith('.pkl') else f
        if norm_alnum(base) == target_norm:
            return os.path.join(MODELS_DIR, f), expected_filename, f

    # Fallback: startswith match
    for f in candidates:
        if f.lower().startswith(expected.lower()):
            return os.path.join(MODELS_DIR, f), expected_filename, f

    return None, expected_filename, None

@app.route("/models", methods=["GET"])
def list_models():
    """List all model files in the models directory."""
    if not os.path.exists(MODELS_DIR):
        return jsonify({"models": []})
    files = sorted(f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".pkl"))
    return jsonify({"models": files})

@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.get_json(force=True)
    country = data.get("country")
    zone = data.get("zone")
    technology = data.get("technology")
    kpi = data.get("kpi")
    input_data = data.get("data")

    if not all([country, zone, technology, kpi, input_data]):
        return jsonify({"error": "Missing required fields"}), 400

    model_path, expected_filename, relaxed_match = resolve_model_path(country, zone, technology, kpi)

    if not model_path:
        return jsonify({
            "error": f"Model not found: {expected_filename}",
            "available_models": os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
        }), 404

    if relaxed_match:
        logger.warning("Using relaxed match file %s for requested model %s", relaxed_match, expected_filename)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error("Error loading model %s: %s", model_path, e)
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    try:
        df = pd.DataFrame(input_data)
        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        logger.error("Error making prediction: %s", e)
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.route("/debug/resolve_model", methods=["GET", "POST"])
def debug_resolve_model():
    """Debug endpoint to show resolved model path without prediction."""
    if request.method == "POST":
        data = request.get_json(force=True)
    else:  # GET request — read from query parameters
        data = {
            "country": request.args.get("country"),
            "zone": request.args.get("zone"),
            "technology": request.args.get("technology"),
            "kpi": request.args.get("kpi"),
        }

    country = data.get("country")
    zone = data.get("zone")
    technology = data.get("technology")
    kpi = data.get("kpi")

    if not all([country, zone, technology, kpi]):
        return jsonify({"error": "Missing required fields"}), 400

    model_path, expected_filename, relaxed_match = resolve_model_path(country, zone, technology, kpi)
    return jsonify({
        "expected_filename": expected_filename,
        "resolved_path": model_path,
        "relaxed_match": relaxed_match,
        "available_models": os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
    })

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Forecast API is running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
