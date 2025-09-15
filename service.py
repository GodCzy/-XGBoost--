"""Flask API service for emission prediction."""

from __future__ import annotations

import logging
from flask import Flask, jsonify, render_template, request
import pandas as pd

from config import Config
from data_preprocessing import clean_data, engineer_features, preprocess
from emission_predictor import ModelManager
from main import generate_synthetic_data

config = Config()
logging.basicConfig(
    level=config.logging_level(),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """Serve the interactive dashboard."""
    return render_template("index.html")


logger.info("Training model for API service")
train_data = generate_synthetic_data()
X_train, y_train, scaler, _ = preprocess(train_data)
model = ModelManager(seed=42)
model.train(X_train, y_train)


@app.route("/predict", methods=["POST"])
def predict():
    """Return emission predictions for provided features."""
    try:
        payload = request.get_json(force=True)
        df = pd.DataFrame(payload)
        df = clean_data(df)
        df = engineer_features(df)
        X_in = scaler.transform(df)
        preds = model.predict(X_in)
        return jsonify({"prediction": preds.tolist()})
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
