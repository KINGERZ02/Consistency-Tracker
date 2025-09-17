
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance


df = pd.read_csv("habit_tracking_synthetic.csv")

# Convert date → weekday feature
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.day_name()
df['weekday'] = df['weekday'].astype('category').cat.codes  # Encode weekdays


X = df.drop(columns=['date', 'daily_productivity'])
y = df['daily_productivity']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("XGBoost Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")


plt.figure(figsize=(8,5))
plot_importance(model, importance_type="weight")
plt.title("Feature Importance (XGBoost)")
plt.show()
