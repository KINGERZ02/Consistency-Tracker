import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
import shap
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load & Prepare Data

from config import DATA_PATH
df = pd.read_csv(DATA_PATH)

# Convert date → weekday feature
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.day_name()
df['weekday'] = df['weekday'].astype('category').cat.codes  # Encode weekdays

X = df.drop(columns=['date', 'daily_productivity'])
y = df['daily_productivity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train XGBoost Model

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


# Evaluate Performance

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("XGBoost Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")


# XGBoost Feature Importance

plt.figure(figsize=(8, 5))
plot_importance(model, importance_type="weight")
plt.title("Feature Importance (XGBoost)")
plt.show()


# SHAP Explainability

print("\nGenerating SHAP explanations...")

# 1. Initialize SHAP explainer
explainer = shap.Explainer(model, X_train)

# 2. Compute SHAP values for the test set
shap_values = explainer(X_test)

# 3. Global Feature Importance (SHAP bar plot)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# 4. Feature effect distribution (red=high value, blue=low value)
shap.summary_plot(shap_values, X_test)

# 5. Local explanation for the first test sample
shap.plots.waterfall(shap_values[0])
