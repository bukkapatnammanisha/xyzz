import json

# Load logs
with open("logs/prediction_logs.json", "r") as f:
    logs = json.load(f)

# Extract predictions
predictions = [entry["output"]["prediction"] for entry in logs]

avg_prediction = sum(predictions) / len(predictions)

drift_detected = avg_prediction > 140

result = {
    "avg_prediction": avg_prediction,
    "drift_detected": drift_detected
}

# Save result
with open("results/step3_s1.json", "w") as f:
    json.dump(result, f, indent=4)

print("Monitoring done")