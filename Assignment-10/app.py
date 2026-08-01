"""
app.py
Assignment 10 - Task 3: API Development

Flask REST API that loads the trained model and serves heart disease
risk predictions from patient clinical data submitted as JSON.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model and feature order at startup
model = joblib.load("model.pkl")
feature_names = joblib.load("feature_names.pkl")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate that all required fields are present
        missing = [f for f in feature_names if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing required fields: {missing}"
            }), 400

        # Build feature vector in the exact order used during training
        features = np.array([[data[f] for f in feature_names]], dtype=float)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "risk_probability": round(float(probability), 4)
        }), 200

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
