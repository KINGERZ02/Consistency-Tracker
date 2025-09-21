import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Consistency Tracker", layout="wide")
st.title("🔥Consistency Tracker")

# Tabs for sections
tabs = st.tabs(["Daily Log", "Dataset Insights", "Global SHAP"])

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
                st.success(f"Predicted Productivity: {round(result['predicted_productivity'], 2)}")

                # Top 3 SHAP insights (text bullets)
                if result.get("insights"):
                    st.subheader("Insights")
                    for insight in result["insights"]:
                        st.write(f"- {insight}")

                # Full SHAP explanation (table)
                if result.get("shap_explanation"):
                    st.subheader("All Feature Contributions (SHAP)")
                    factors_df = pd.DataFrame(result["shap_explanation"], index=[0]).T
                    factors_df.columns = ["SHAP Value"]
                    st.dataframe(factors_df.sort_values("SHAP Value", key=abs, ascending=False))

            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {e}")


# ---------------- DATASET INSIGHTS ----------------
with tabs[1]:
    st.header("Dataset Insights")

    uploaded_file = st.file_uploader("Upload your habit dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/upload_csv", files={"file": uploaded_file})
            if response.status_code == 200:
                result = response.json()
                st.success("File uploaded successfully!")
                st.json(result)
            else:
                st.error(f"Upload failed: {response.status_code}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

        # Call summary
        st.subheader("Summary")
        try:
            res = requests.get(f"{API_URL}/get_summary")
            st.json(res.json())
        except:
            st.warning("Could not fetch summary.")

        # Call weekly
        st.subheader("Weekly Averages")
        try:
            res = requests.get(f"{API_URL}/get_weekly")
            weekly_data = res.json().get("weekly_average_productivity", {})
            if weekly_data:
                df_weekly = pd.DataFrame.from_dict(weekly_data, orient="index", columns=["Avg Productivity"])
                st.bar_chart(df_weekly)
                st.dataframe(df_weekly)
            else:
                st.json(res.json())
        except:
            st.warning("Could not fetch weekly averages.")

        # Call streaks
        st.subheader("Streaks")
        try:
            res = requests.get(f"{API_URL}/get_streaks")
            st.json(res.json())
        except:
            st.warning("Could not fetch streak info.")


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
