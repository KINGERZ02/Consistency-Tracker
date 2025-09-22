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

# ---------------- DATASET INSIGHTS ----------------
with tabs[1]:
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    import calplot

    st.header("Dataset Insights")

    uploaded_file = st.file_uploader("Upload your habit dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        # show filename
        st.markdown(f"**File:** {uploaded_file.name}")

        # 1) POST file to backend /upload_csv
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            resp = requests.post(f"{API_URL}/upload_csv", files=files, timeout=60)
        except Exception as e:
            st.error(f"Upload failed: {e}")
            resp = None

        if resp is None:
            st.stop()

        if resp.status_code != 200:
            st.error(f"Upload failed (status {resp.status_code}): {resp.text}")
        else:
            upload_info = resp.json()

            # --- KPIs Row ---
            cols = st.columns(3)
            cols[0].metric("Rows", int(upload_info.get("rows_loaded", 0)))
            cols[1].metric("Avg Actual Prod", float(upload_info.get("average_actual_productivity", 0)))
            cols[2].metric("Avg Predicted Prod", float(upload_info.get("average_predicted_productivity", 0)))

            # --- Calendar Streak Tracker ---
            st.subheader("🔥 Calendar Streak Tracker (last 3 months)")
            try:
                uploaded_file.seek(0)
                local_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
                if "date" in local_df.columns:
                    local_df["date"] = pd.to_datetime(local_df["date"], errors="coerce")
                if "daily_productivity" in local_df.columns:
                    prod_series = local_df.set_index("date")["daily_productivity"]
                    last_date = prod_series.index.max()
                    three_months_back = last_date - pd.DateOffset(months=3)
                    prod_series = prod_series[prod_series.index >= three_months_back]

                    fig, ax = calplot.calplot(
                        prod_series,
                        cmap="Oranges",
                        edgecolor="#888888",
                        colorbar=True,
                        linewidth=0.5,
                        fillcolor="black"  # blank days = black
                    )
                    st.pyplot(fig, transparent=True)
            except Exception as e:
                st.warning(f"Could not generate calendar: {e}")

            st.markdown("---")

            # --- Backend Summary ---
            st.subheader("Summary")
            try:
                s = requests.get(f"{API_URL}/get_summary", timeout=20)
                summary = s.json() if s.status_code == 200 else {}
            except Exception as e:
                summary = {}

            if summary:
                cols = st.columns(len(summary))
                for i, (k, v) in enumerate(summary.items()):
                    cols[i].metric(k.replace("_", " ").title(), round(v, 2) if isinstance(v, (int, float)) else v)
            else:
                st.info("No summary data available.")

            st.markdown("---")

            # --- Daily productivity time series ---
            st.subheader("Daily Productivity (actual)")
            if "daily_productivity" in local_df.columns and "date" in local_df.columns:
                ts_df = local_df.sort_values("date")
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("none")
                ax.plot(ts_df["date"], ts_df["daily_productivity"], marker='o', linewidth=1.5, color="#FF6B00")
                ax.set_xlabel("")
                ax.set_ylabel("Actual Productivity", color="#888888")
                ax.tick_params(colors="#888888")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                st.pyplot(fig, transparent=True)

            st.markdown("---")

            # --- Weekly actual vs predicted ---
            st.subheader("Weekly Averages (actual vs predicted)")
            try:
                w = requests.get(f"{API_URL}/get_weekly", timeout=20)
                weekly_pred = w.json().get("weekly_average_productivity", {})
            except Exception:
                weekly_pred = {}

            weekly_actual = {}
            if "weekday_name" not in local_df.columns:
                local_df["weekday_name"] = local_df["date"].dt.day_name()
            weekly_actual_series = local_df.groupby("weekday_name")["daily_productivity"].mean().dropna()
            weekly_actual = weekly_actual_series.to_dict()

            wd_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            pred_vals, actual_vals, labels = [], [], []
            for i, name in enumerate(wd_order):
                labels.append(name[:3])
                val_pred = weekly_pred.get(str(i)) or weekly_pred.get(i) or weekly_pred.get(name)
                pred_vals.append(float(val_pred) if val_pred is not None else np.nan)
                val_act = weekly_actual.get(name, np.nan)
                actual_vals.append(float(val_act) if not pd.isna(val_act) else np.nan)

            x = np.arange(len(labels))
            width = 0.35
            fig, ax = plt.subplots(figsize=(9, 3))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            ax.bar(x - width/2, actual_vals, width, label="Actual", color="#666666", alpha=0.9)
            ax.bar(x + width/2, pred_vals, width, label="Predicted", color="#FF6B00", alpha=0.95)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, color="#888888")
            ax.set_ylabel("Avg Productivity", color="#888888")
            ax.tick_params(colors="#888888")
            ax.legend(frameon=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig, transparent=True)

            st.markdown("---")

            # --- Streaks (backend + strip visualization) ---
            st.subheader("Streaks")
            try:
                r = requests.get(f"{API_URL}/get_streaks", timeout=20)
                streak_info = r.json()

                # Display key streak stats as metrics
                cols = st.columns(3)
                cols[0].metric("🔥 Current Streak", streak_info.get("current_streak", 0))
                cols[1].metric("🏆 Longest Streak", streak_info.get("longest_streak", 0))
                cols[2].metric("Threshold", streak_info.get("threshold", 7))

                # Boolean streak strip visualization
                thr = streak_info.get("threshold", 7)
                bool_series = (local_df["daily_productivity"] >= thr).astype(int).reset_index(drop=True)
                fig, ax = plt.subplots(figsize=(10,1.2))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("none")
                ax.imshow([bool_series.values], aspect="auto", cmap=plt.get_cmap("Oranges"))

                # Remove ticks and spines, grey styling
                ax.set_yticks([])
                ax.set_xticks([])
                ax.tick_params(colors="#888888")
                for spine in ax.spines.values():
                    spine.set_visible(False)

                st.pyplot(fig, transparent=True)
            except Exception as e:
                st.warning(f"Could not fetch streaks: {e}")
