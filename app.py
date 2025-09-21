import streamlit as st
import requests
import pandas as pd
import shap
import numpy as np

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Consistency Tracker", layout="wide")
st.title("🔥Consistency Tracker")

# Tabs for sections
tabs = st.tabs(["Daily Log", "Dataset Insights", "Global SHAP"])

# ---------------- DAILY LOG ----------------
import numpy as np
import shap
import matplotlib.pyplot as plt

# ---------------- DAILY LOG ----------------
with tabs[0]:
    st.header("Daily Productivity Prediction")

    with st.form("daily_log_form"):
        # Binary tasks (0 or 1)
        leetcode = 1 if st.checkbox("LeetCode Practice") else 0
        capstone = 1 if st.checkbox("Capstone Work") else 0
        projects = 1 if st.checkbox("Side Projects") else 0
        misc = 1 if st.checkbox("Miscellaneous Tasks") else 0

        # Continuous values
        sleep_hours = st.slider("Sleep Hours", 0, 10, 7)
        sleep_quality = st.slider("Sleep Quality", 0, 10, 7)
        mood = st.slider("Mood", 0, 10, 6)
        stress = st.slider("Stress", 0, 10, 4)
        energy = st.slider("Energy", 0, 10, 6)
        weekday = st.selectbox(
            "Weekday",
            list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "leetcode": leetcode, "capstone": capstone,
            "projects": projects, "misc": misc,
            "sleep_hours": sleep_hours, "sleep_quality": sleep_quality,
            "mood": mood, "stress": stress, "energy": energy,
            "weekday": weekday
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()

                # Prediction
                prediction = result['predicted_productivity']
                st.success(f"Predicted Productivity: {round(prediction, 2)}")

                # Motivational Insight
                if result.get("insights"):
                    st.subheader("Coach's Insight")
                    for insight in result["insights"]:
                        st.write(insight)

                # Waterfall Plot (Visual SHAP)
                if result.get("shap_explanation"):
                    st.subheader("Visual Explanation (Waterfall)")

                    values = np.array(list(result["shap_explanation"].values()))
                    features = list(result["shap_explanation"].keys())
                    base_value = prediction - np.sum(values)

                    shap_values = shap.Explanation(
                        values=values,
                        base_values=base_value,
                        data=features
                    )

                    # Create matplotlib figure for Streamlit
                    fig, ax = plt.subplots(figsize=(8, 5))

                    # Transparent floating background
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor("none")

                    # Waterfall plot
                    shap.plots.waterfall(shap_values, show=False)

                    # Apply fire palette colors
                    for patch, val in zip(ax.patches, values):
                        patch.set_color("#FF6B00" if val > 0 else "#333333")

                    # Remove clutter
                    for txt in ax.texts:
                        txt.set_visible(False)
                    ax.grid(False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    st.pyplot(fig, transparent=True)

            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {e}")


# ---------------- GLOBAL SHAP ----------------
with tabs[2]:
    st.header("Global SHAP Feature Importance")

    try:
        res = requests.get(f"{API_URL}/get_top_features")
        if res.status_code == 200:
            top_feats = res.json().get("top_features", [])
            if top_feats:
                df_feats = pd.DataFrame(top_feats)
                st.bar_chart(df_feats.set_index("feature"))
                st.dataframe(df_feats)
            else:
                st.warning("No SHAP features found.")
        else:
            st.error(f"Error: {res.status_code}")
    except Exception as e:
        st.error(f"Could not fetch SHAP features: {e}")
