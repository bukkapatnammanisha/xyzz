from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# -------------------------
# Train model (simple)
# -------------------------
df = pd.read_csv("data/training_data.csv")

X = df.drop("ball_speed_kmph", axis=1)
y = df["ball_speed_kmph"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# -------------------------
# Input schema (FIXED)
# -------------------------
class InputData(BaseModel):
    bowler_id: int
    match_id: int
    overs_bowled: int
    runs_conceded: int
    wickets_taken: int
    pitch_condition: int

# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/forecast")
def predict(data: InputData):
    try:
        features = np.array([[
            data.bowler_id,
            data.match_id,
            data.overs_bowled,
            data.runs_conceded,
            data.wickets_taken,
            data.pitch_condition
        ]])

        prediction = model.predict(features)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=422, detail="Invalid input")