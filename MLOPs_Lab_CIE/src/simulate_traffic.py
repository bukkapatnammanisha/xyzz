import requests
import json
import time

url = "http://127.0.0.1:8501/forecast"

data_samples = [
    {
        "bowler_id": 201,
        "match_id": 5,
        "overs_bowled": 4,
        "runs_conceded": 30,
        "wickets_taken": 2,
        "pitch_condition": 1
    },
    {
        "bowler_id": 202,
        "match_id": 5,
        "overs_bowled": 4,
        "runs_conceded": 45,
        "wickets_taken": 0,
        "pitch_condition": 3
    },
    {
        "bowler_id": 203,
        "match_id": 6,
        "overs_bowled": 4,
        "runs_conceded": 25,
        "wickets_taken": 3,
        "pitch_condition": 1
    }
]

logs = []

for data in data_samples:
    response = requests.post(url, json=data)
    result = response.json()
    
    logs.append({
        "input": data,
        "output": result
    })

    print(result)
    time.sleep(1)

# Save logs
with open("logs/prediction_logs.json", "w") as f:
    json.dump(logs, f, indent=4)

print("Traffic simulation done")