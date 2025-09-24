import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shap
import numpy as np

# --- Init app ---
app = FastAPI(title="Consistency Tracker API")

# --- Load model ---
MODEL_PATH = "src/models/xgb_model.pkl"
EXPLAINER_PATH = "src/models/shap_explainer.pkl"

# Try loading model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Could not load model: {e}")

# Try loading explainer, else build a new one
try:
    explainer = joblib.load(EXPLAINER_PATH)
    print("✅ Loaded saved SHAP explainer")
except Exception as e:
    print(f"⚠️ Could not load saved explainer ({e}), building new TreeExplainer...")
    explainer = shap.TreeExplainer(model)

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

# --- Predict (daily log) ---
@app.post("/predict")
def predict(habit: HabitLog):
    import numpy as np
    try:
        input_data = habit.dict()
        df_input = pd.DataFrame([input_data])

        # Make prediction and clamp to [0, 10]
        prediction = model.predict(df_input)[0]
        prediction = float(np.clip(prediction, 0, 10))  # force plain float

        # Check if no work was done
        no_work = all(input_data[task] == 0 for task in ["leetcode", "capstone", "projects", "misc"])

        # Compute SHAP values safely
        shap_values = explainer(df_input)
        feature_contribs = {
            k: float(v) for k, v in zip(input_data.keys(), shap_values.values[0])  # force floats
        }

        insights = []

        if no_work:
            insights.append(
                "You didn’t complete any tasks today, so productivity stayed minimal. "
                "But you managed good sleep and energy!! that’s a strong base. "
                "Tomorrow, try adding even one task to keep the momentum going."
            )
        else:
            sorted_contribs = sorted(feature_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
            top_factors = sorted_contribs[:3]

            positives, negatives = [], []
            for feat, val in top_factors:
                feat_name = feat.replace("_", " ")
                if val > 0:
                    positives.append(feat_name)
                else:
                    negatives.append(feat_name)

            sentence = ""
            if positives:
                sentence += "Great job!! " + " and ".join(positives) + " boosted your productivity today. "
            if negatives:
                if positives:
                    sentence += "But " + " and ".join(negatives) + " held you back a little. "
                else:
                    sentence += "Productivity dipped mainly due to " + " and ".join(negatives) + ". "
            sentence += "Keep building consistency!! even small steps daily will compound!"
            insights.append(sentence.strip())

        return {
            "predicted_productivity": prediction,
            "shap_explanation": feature_contribs,
            "insights": insights
        }

    except Exception as e:
        import traceback
        print("❌ Prediction error:", traceback.format_exc())
        # Always return fallback response
        return {
            "predicted_productivity": 0.0,
            "shap_explanation": {},
            "insights": [f"⚠️ Something went wrong: {str(e)}"]
        }



# --- Upload CSV / Excel (batch analysis) ---
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept CSV or Excel file, ensure date -> weekday, validate feature columns,
    generate predictions, and if daily_productivity is missing, fill it with predictions.
    Stores dataframe in global `uploaded_df`.
    Returns JSON-safe summary including whether actual daily_productivity was used.
    """
    global uploaded_df
    import io, os

    # read bytes
    contents = await file.read()
    fname = getattr(file, "filename", "uploaded_file")
    ext = os.path.splitext(fname)[1].lower()

    # Load file into pandas DataFrame
    try:
        if ext in [".csv", ""]:
            # default to csv if extension missing
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file format. Please upload CSV or Excel (.xls/.xlsx)."}
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    # Required features for model
    feature_cols = [
        "leetcode", "capstone", "projects", "misc",
        "sleep_hours", "sleep_quality", "mood", "stress",
        "energy", "weekday"
    ]

    # 1) Ensure 'date' column exists and parse it
    if "date" not in df.columns:
        return {"error": "File must include a 'date' column (YYYY-MM-DD or similar)."}
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception as e:
        return {"error": f"Could not parse 'date' column: {e}"}

    # 2) Create numeric weekday (0=Mon .. 6=Sun). Fill missing dates with 0.
    # Using dt.weekday gives consistent numeric mapping expected by model.
    df["weekday"] = df["date"].dt.weekday
    df["weekday"] = df["weekday"].fillna(0).astype(int)

    # 3) Validate feature columns (excluding 'weekday' because we just added it)
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        return {"error": f"Missing required feature columns: {missing_features}"}

    # 4) Ensure feature columns are numeric where appropriate
    for col in ["leetcode", "capstone", "projects", "misc",
                "sleep_hours", "sleep_quality", "mood", "stress",
                "energy", "weekday"]:
        # coerce to numeric where possible
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Drop rows that cannot be used for prediction (if any features are NaN)
    # but keep track of original length
    n_original = len(df)
    usable_mask = df[feature_cols].notna().all(axis=1)
    n_usable = int(usable_mask.sum())
    if n_usable == 0:
        return {"error": "No rows contain all required feature values for prediction."}

    # Prepare prediction input using only usable rows (but attach predictions back to full df)
    try:
        preds = model.predict(df.loc[usable_mask, feature_cols])
    except Exception as e:
        import traceback
        print("❌ Prediction generation error:", traceback.format_exc())
        return {"error": f"Model prediction failed: {e}"}

    # Assign predictions as Python floats
    preds = [float(x) for x in preds]
    df.loc[usable_mask, "predicted_productivity"] = preds

    # 6) Decide whether to use provided 'daily_productivity'
    used_actuals = False
    if "daily_productivity" in df.columns:
        # coerce to numeric
        df["daily_productivity"] = pd.to_numeric(df["daily_productivity"], errors="coerce")
        # if at least one non-null actual exists, we'll treat actuals as provided
        if df["daily_productivity"].notna().any():
            used_actuals = True
            # For rows with missing actuals, fallback to predicted
            df["daily_productivity"] = df["daily_productivity"].fillna(df["predicted_productivity"])
        else:
            # column exists but all NaN -> treat as missing and replace with predicted
            df["daily_productivity"] = df["predicted_productivity"]
            used_actuals = False
    else:
        # No actuals provided -> use predictions as daily_productivity
        df["daily_productivity"] = df["predicted_productivity"]
        used_actuals = False

    # 7) Store globally (convert numeric columns to native types where needed)
    # but keep DataFrame object for subsequent endpoints
    uploaded_df = df.copy()

    # 8) Compute safe summary stats for response (convert numpy types to native)
    avg_actual = None
    if uploaded_df["daily_productivity"].notna().any():
        avg_actual = float(uploaded_df["daily_productivity"].mean())

    avg_pred = None
    if uploaded_df["predicted_productivity"].notna().any():
        avg_pred = float(uploaded_df["predicted_productivity"].mean())

    return {
        "filename": str(fname),
        "rows_loaded": int(n_original),
        "rows_usable_for_prediction": int(n_usable),
        "used_actuals": bool(used_actuals),
        "average_actual_productivity": round(avg_actual, 2) if avg_actual is not None else None,
        "average_predicted_productivity": round(avg_pred, 2) if avg_pred is not None else None,
        "message": "File processed. If daily_productivity was missing it has been filled with model predictions."
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
@app.get("/get_global_shap")
def get_global_shap(sample_size: int = 200):
    global uploaded_df
    if uploaded_df is None:
        return {"error": "No data uploaded yet. Please upload a CSV first."}

    try:
        feature_cols = ["leetcode","capstone","projects","misc",
                        "sleep_hours","sleep_quality","mood","stress",
                        "energy","weekday"]

        # Subsample for performance
        n_rows = min(sample_size, len(uploaded_df))
        sample_df = uploaded_df[feature_cols].sample(n_rows, random_state=42)

        # Compute SHAP values
        shap_values = explainer(sample_df)

        # Defensive handling for different SHAP versions
        shap_arr = getattr(shap_values, "values", None)
        if shap_arr is None:
            shap_arr = shap_values  # some versions return ndarray directly

        shap_arr = np.array(shap_arr)

        # If still invalid
        if shap_arr.size == 0:
            return {"error": "⚠️ SHAP values came out empty. Check explainer setup."}

        # Global importance
        mean_abs = np.abs(shap_arr).mean(axis=0)
        feature_importance = [
            {"feature": f, "importance": float(round(v, 4))}
            for f, v in zip(feature_cols, mean_abs)
        ]
        feature_importance = sorted(feature_importance, key=lambda x: x["importance"], reverse=True)

        # Handle base_value safely
        base_value = shap_values.base_values
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.mean(base_value))
        else:
            base_value = float(base_value)

        return {
            "feature_importance": feature_importance,
            "shap_sample": {
                feature_cols[i]: shap_arr[:, i].tolist()
                for i in range(len(feature_cols))
            },
            "base_value": base_value
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Global SHAP computation failed: {e}"}
