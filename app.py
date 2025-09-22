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

    # --- Helper: Calculate streaks locally ---
    def calculate_streaks(series, threshold=7):
        streaks = (series >= threshold).astype(int)
        longest = 0
        current = 0
        temp = 0
        for val in streaks:
            if val == 1:
                temp += 1
                longest = max(longest, temp)
            else:
                temp = 0
        current = 0
        for val in reversed(streaks):
            if val == 1:
                current += 1
            else:
                break
        return current, longest

    uploaded_file = st.file_uploader("Upload your habit dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        st.markdown(f"**File:** {uploaded_file.name}")

        # Send file to backend
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

                    fig, axes = calplot.calplot(
                        prod_series,
                        cmap="Oranges",
                        edgecolor="#888888",
                        colorbar=True,
                        linewidth=0.5,
                        fillcolor="black"  # blank days = black
                    )

                    # Recolor all text/ticks grey
                    if isinstance(axes, np.ndarray):
                        axs = axes.flatten()
                    else:
                        axs = [axes]
                    for ax in axs:
                        for text in ax.texts:
                            text.set_color("#888888")
                        ax.tick_params(colors="#888888")
                        if ax.yaxis.label: ax.yaxis.label.set_color("#888888")
                        if ax.xaxis.label: ax.xaxis.label.set_color("#888888")
                    if fig.axes:
                        for a in fig.axes:
                            a.tick_params(colors="#888888")
                            if a.yaxis.label: a.yaxis.label.set_color("#888888")
                            if a.xaxis.label: a.xaxis.label.set_color("#888888")

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

            # --- Daily Productivity Time Series ---
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

            # --- Weekly Actual vs Predicted ---
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

            # --- Streaks (local calculation) ---
            st.subheader("Streaks")
            try:
                thr = 7
                if "daily_productivity" in local_df.columns:
                    series = local_df["daily_productivity"]
                    current_streak, longest_streak = calculate_streaks(series, thr)

                    cols = st.columns(3)
                    cols[0].metric("🔥 Current Streak", current_streak)
                    cols[1].metric("🏆 Longest Streak", longest_streak)
                    cols[2].metric("Threshold", thr)

                    bool_series = (series >= thr).astype(int).reset_index(drop=True)
                    fig, ax = plt.subplots(figsize=(10,1.2))
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor("none")
                    ax.imshow([bool_series.values], aspect="auto", cmap=plt.get_cmap("Oranges"))
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.tick_params(colors="#888888")
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    st.pyplot(fig, transparent=True)
            except Exception as e:
                st.warning(f"Could not compute streaks: {e}")

# ---------------- GLOBAL SHAP ----------------
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global SHAP tab ---
def global_shap_tab():
    st.subheader("🌍 Global SHAP Insights")

    # Call FastAPI
    try:
        resp = requests.get("http://localhost:8000/get_global_shap")  # change if deployed
        data = resp.json()
    except Exception as e:
        st.error(f"❌ Could not fetch SHAP data: {e}")
        return

    if "error" in data:
        st.warning(data["error"])
        return

    # --- Feature Importance (Bar Plot) ---
    st.markdown("#### 🔥 Feature Importance (mean |SHAP|)")
    fi_df = pd.DataFrame(data["feature_importance"])

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=fi_df,
        x="importance", y="feature",
        ax=ax, palette=["#FF6600"] * len(fi_df)  # fire orange
    )
    ax.set_xlabel("Mean |SHAP Value|", color="#888888")
    ax.set_ylabel("Feature", color="#888888")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
    st.pyplot(fig)

    # --- Beeswarm Plot ---
    st.markdown("#### 🐝 Beeswarm Distribution (SHAP effects)")
    shap_sample = pd.DataFrame(data["shap_sample"])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.stripplot(
        data=shap_sample,
        orient="h",
        size=4,
        palette=["#FF6600"],
        alpha=0.6
    )
    ax.set_xlabel("SHAP Value (Impact on Productivity)", color="#888888")
    ax.set_ylabel("Features", color="#888888")
    ax.set_yticks(range(len(shap_sample.columns)))
    ax.set_yticklabels(shap_sample.columns, color="#888888")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
    st.pyplot(fig)

    st.info("Orange → stronger influence. Values spread left/right → how each feature pushes productivity up/down.")

