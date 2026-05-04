import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("ball_speed_kmph", axis=1)
y = df["ball_speed_kmph"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow experiment
mlflow.set_experiment("ultratrack-ball-speed-kmph")

results = []

models = {
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log params + metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("project_phase", "model_selection")

        # ✅ IMPORTANT (for Task 4)
        mlflow.sklearn.log_model(model, "model")

        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse)
        })

# Select best model
best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "ultratrack-ball-speed-kmph",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

# Save JSON
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 Done")