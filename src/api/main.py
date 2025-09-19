import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- Init FastAPI app ---
app = FastAPI(title="Consistency Tracker API")

# --- Load model + SHAP explainer ---
MODEL_PATH = "src/models/xgb_model.pkl"
EXPLAINER_PATH = "src/models/shap_explainer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Could not load model/explainer: {e}")

# --- Input schema ---
class HabitLog(BaseModel):
    leetcode: float
    capstone: float
    projects: float
    misc: float
    sleep_hours: float
    sleep_quality: float
    mood: float
    stress: float
    energy: float
    weekday: int  # 👈 include weekday because your model expects it

# --- Healthcheck ---
@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "message": "Consistency Tracker API is live "}

# --- Predict ---
@app.post("/predict")
def predict(log: HabitLog):
    input_df = pd.DataFrame([log.dict()])

    # Predict
    prediction = model.predict(input_df)[0]

    # SHAP values
    shap_values = explainer(input_df)
    feature_contribs = dict(zip(input_df.columns, shap_values.values[0].tolist()))

    # Top 3 insights
    top_factors = sorted(feature_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    insights = [f"{feat} contributed {round(val,3)}" for feat, val in top_factors]

    return {
        "predicted_productivity": float(prediction),
        "shap_explanation": feature_contribs,
        "insights": insights
    }
