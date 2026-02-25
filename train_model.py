import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
data = pd.read_csv("sleep_data.csv")

X = data[["temperature", "light", "noise", "heart_rate"]]
y_time = data["sleep_time"]
y_quality = data["sleep_quality"]

# Train models
time_model = RandomForestRegressor()
quality_model = RandomForestRegressor()

time_model.fit(X, y_time)
quality_model.fit(X, y_quality)

# Save models
joblib.dump(time_model, "sleep_time_model.pkl")
joblib.dump(quality_model, "sleep_quality_model.pkl")

print("Models trained and saved successfully")