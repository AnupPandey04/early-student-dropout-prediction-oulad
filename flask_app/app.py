from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "xgb_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "model", "feature_columns.pkl")

# Load model and features
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():

    # Collect input
    input_data = {
        "gender": request.form["gender"],
        "age_band": request.form["age_band"],
        "highest_education": request.form["highest_education"],
        "disability": request.form["disability"],
        "total_clicks": float(request.form["total_clicks"]),
        "active_days": int(request.form["active_days"]),
        "unique_activities": int(request.form["unique_activities"]),
        "avg_clicks_per_day": float(request.form["avg_clicks_per_day"])
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # One-hot encode
    df_encoded = pd.get_dummies(df)

    # Align columns
    df_encoded = df_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict probability
    prob = model.predict_proba(df_encoded)[0][1]
    prob = round(float(prob), 2)

    # Risk level
    if prob >= 0.6:
        risk = "High"
    elif prob >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return render_template(
        "index.html",
        prediction=prob,
        risk_level=risk
    )

if __name__ == "__main__":
    app.run(debug=True)
