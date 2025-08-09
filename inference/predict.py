import os
import joblib

# Flask API for SageMaker
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# SageMaker model directory
MODEL_PATH = os.path.join("/opt/ml/model", "model.joblib")

# Fallback for local testing
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "model.joblib"

# Load model once when container starts
model = joblib.load(MODEL_PATH)

@app.route("/ping", methods=["GET"])
def ping():
    # Health check endpoint
    return jsonify(status="ok"), 200

@app.route("/invocations", methods=["POST"])
def invocations():
    # Parse input JSON
    data = request.get_json()
    features = np.array(data["instances"])
    # Predict
    preds = model.predict(features)
    # Return predictions
    return jsonify(predictions=preds.tolist())

# Local debug
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
