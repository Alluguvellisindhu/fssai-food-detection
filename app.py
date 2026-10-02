from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "food_safety_model.pkl"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            data = {
                "food_category": request.form["food_category"],
                "moisture_percent": float(request.form["moisture_percent"]),
                "ph": float(request.form["ph"]),
                "protein_g_100g": float(request.form["protein_g_100g"]),
                "fat_g_100g": float(request.form["fat_g_100g"]),
                "bacterial_count_cfu_g": float(request.form["bacterial_count_cfu_g"]),
                "contaminant_level_mg_kg": float(request.form["contaminant_level_mg_kg"]),
                "storage_temperature_c": float(request.form["storage_temperature_c"]),
                "storage_days": int(request.form["storage_days"]),
            }

            input_df = pd.DataFrame([data])
            prediction = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                if "At Risk" in classes:
                    probability = round(
                        probabilities[classes.index("At Risk")] * 100, 2
                    )

        except (ValueError, KeyError):
            error = "Please enter valid values in all fields."

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
