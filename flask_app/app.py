from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

#Validation Helper
def safe_number(value, min_val, max_val, cast_type=int):
    try:
        value = cast_type(value)
        if value < min_val or value > max_val:
            raise ValueError
        return value
    except:
        raise ValueError(f"Value must be between {min_val} and {max_val}")


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
    try:
        input_data = {
            "gender": request.form["gender"],
            "age_band": request.form["age_band"],
            "highest_education": request.form["highest_education"],
            "disability": request.form["disability"],

            # Dataset-aware validation (OULAD – first 4 weeks)
            "total_clicks": safe_number(request.form["total_clicks"], 0, 500, float),
            "active_days": safe_number(request.form["active_days"], 0, 28, int),
            "unique_activities": safe_number(request.form["unique_activities"], 0, 50, int),
            "avg_clicks_per_day": safe_number(request.form["avg_clicks_per_day"], 0, 50, float),
        }
    except ValueError as e:
        return render_template(
            "index.html",
            error=str(e),
            prediction=None
        )


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
    app.run()
