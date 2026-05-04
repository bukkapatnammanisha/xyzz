import mlflow
import mlflow.sklearn
import json

# Set experiment
mlflow.set_experiment("ultratrack-ball-speed-kmph")

# Get latest run
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("ultratrack-ball-speed-kmph")

runs = client.search_runs(experiment.experiment_id)

# Take best run (first one)
run = runs[0]
run_id = run.info.run_id

# Register model
model_uri = f"runs:/{run_id}/model"

model_name = "ultratrack-ball-speed-kmph-predictor"

result = mlflow.register_model(model_uri, model_name)

# Get version
version = result.version

# Get RMSE
rmse = run.data.metrics.get("rmse", 0.0)

# Save JSON
output = {
    "registered_model_name": model_name,
    "version": int(version),
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": rmse
}

with open("results/step4_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Model registered successfully")