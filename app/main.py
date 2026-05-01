from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI()

# ----------------------------
# Load model safely
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "salary_pipeline.pkl")

model = joblib.load(MODEL_PATH)

# ----------------------------
# Input schema (validated)
# ----------------------------
class InputData(BaseModel):
    age: float = Field(..., gt=0, lt=100)
    gender: str
    education: str
    job: str
    experience: float = Field(..., ge=0, lt=50)

# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def home():
    return {"status": "API is running"}

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(data: InputData):

    try:
        input_df = pd.DataFrame([{
            "Age": data.age,
            "Gender": data.gender,
            "Education Level": data.education,
            "Job Title": data.job,
            "Years of Experience": data.experience
        }])

        prediction = model.predict(input_df)[0]

        return {
            "predicted_salary": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))