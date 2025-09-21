import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

# --- Init app ---
app = FastAPI(title="Consistency Tracker API")

# --- Load model + SHAP explainer ---
MODEL_PATH = "src/models/xgb_model.pkl"
EXPLAINER_PATH = "src/models/shap_explainer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Could not load model/explainer: {e}")

# --- Store uploaded data globally (for summary endpoints) ---
uploaded_df = None

# --- Schema for daily log ---
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
    weekday: int


# --- Root ---
@app.get("/")
def root():
    return {"message": "Welcome to Consistency Tracker API 🚀"}


# --- Healthcheck ---
@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "message": "API is live"}


# --- Predict (daily log) ---
@app.post("/predict")
def predict(habit: HabitLog):
    input_data = habit.dict()

    # Make prediction
    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)[0]

    # Compute SHAP values
    shap_values = explainer(df_input)
    feature_contribs = dict(zip(input_data.keys(), shap_values.values[0]))

    # Feature directions (which side is good)
    feature_directions = {
        "sleep_hours": "higher is better",
        "sleep_quality": "higher is better",
        "energy": "higher is better",
        "mood": "higher is better",
        "leetcode": "higher is better",
        "capstone": "higher is better",
        "projects": "higher is better",
        "misc": "higher is better",
        "stress": "lower is better"
    }

    # Sort contributions
    sorted_contribs = sorted(feature_contribs.items(), key=lambda x: abs(x[1]), reverse=True)

    # Take top 3 impactful features
    top_factors = sorted_contribs[:3]
    positives, negatives = [], []

    for feat, val in top_factors:
        direction = feature_directions.get(feat, "higher is better")
        feat_name = feat.replace("_", " ")

        # Positive influence
        if (val > 0 and direction == "higher is better") or (val < 0 and direction == "lower is better"):
            positives.append(feat_name)
        # Negative influence
        else:
            negatives.append(feat_name)

    # Build human-readable sentence
    sentence = ""
    if positives:
        sentence += "Your productivity increased because " + " and ".join(positives) + " were strong. "
    if negatives:
        if positives:
            sentence += "However, it dropped a bit due to " + " and ".join(negatives) + "."
        else:
            sentence += "Your productivity dropped due to " + " and ".join(negatives) + "."

    return {
        "predicted_productivity": float(prediction),
        "shap_explanation": feature_contribs,
        "insights": [sentence.strip()]
    }



# --- Upload CSV (batch analysis) ---
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global uploaded_df
    contents = await file.read()
    import io
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # 1. Convert date → weekday
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name().astype("category").cat.codes

    # 2. Features for prediction
    feature_cols = ["leetcode","capstone","projects","misc",
                    "sleep_hours","sleep_quality","mood","stress",
                    "energy","weekday"]

    # 3. Generate predictions
    preds = model.predict(df[feature_cols])
    df["predicted_productivity"] = preds.astype(float)  # ensure native floats

    # 4. Store for later endpoints
    uploaded_df = df

    return {
        "filename": str(file.filename),
        "rows_loaded": int(len(df)),
        "average_actual_productivity": float(round(df["daily_productivity"].mean(), 2)),
        "average_predicted_productivity": float(round(df["predicted_productivity"].mean(), 2))
    }



@app.get("/get_summary")
def get_summary():
    global uploaded_df
    if uploaded_df is None:
        return {"error": "No data uploaded yet. Please upload a CSV first."}

    avg_prod = float(uploaded_df["predicted_productivity"].mean())
    max_prod = float(uploaded_df["predicted_productivity"].max())
    min_prod = float(uploaded_df["predicted_productivity"].min())

    return {
        "average_productivity": round(avg_prod, 2),
        "max_productivity": round(max_prod, 2),
        "min_productivity": round(min_prod, 2),
        "rows_analyzed": int(len(uploaded_df))
    }



@app.get("/get_weekly")
def get_weekly():
    global uploaded_df
    if uploaded_df is None:
        return {"error": "No data uploaded yet. Please upload a CSV first."}

    weekly_avg = uploaded_df.groupby("weekday")["predicted_productivity"].mean().to_dict()

    # Convert all numpy.float to Python float
    weekly_avg = {str(k): float(v) for k, v in weekly_avg.items()}

    return {"weekly_average_productivity": weekly_avg}



# --- Get Streaks ---
@app.get("/get_streaks")
def get_streaks():
    global uploaded_df
    if uploaded_df is None:
        return {"error": "No data uploaded yet. Please upload a CSV first."}

    threshold = 7
    streak = (uploaded_df["predicted_productivity"] >= threshold).astype(int)
    longest_streak = (streak.groupby((streak != streak.shift()).cumsum())
                      .cumsum().max())

    return {
        "threshold": threshold,
        "longest_high_productivity_streak": int(longest_streak)
    }


# --- Get Top Features ---
@app.get("/get_top_features")
def get_top_features():
    global uploaded_df
    if uploaded_df is None:
        return {"error": "No data uploaded yet. Please upload a CSV first."}

    # Only keep feature columns for SHAP
    feature_cols = ["leetcode","capstone","projects","misc",
                    "sleep_hours","sleep_quality","mood","stress",
                    "energy","weekday"]

    sample_df = uploaded_df[feature_cols].sample(min(50, len(uploaded_df)))
    shap_values = explainer(sample_df)

    feature_importance = dict(
        zip(feature_cols, abs(shap_values.values).mean(0).tolist())
    )

    # Sort and cast to Python float
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "top_features": [
            {"feature": feat, "importance": float(round(val, 3))} for feat, val in top_features
        ]
    }


