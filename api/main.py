from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="No-Show Predictor API")

# Allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
    age: int
    scholarship: int
    hypertension: int
    diabetes: int
    alcoholism: int
    handicap: int
    sms_received: int
    waiting_days: int

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: PatientInput):

    risk_score = 0.5

    return {
        "prediction": "Likely No-Show" if risk_score >= 0.5 else "Likely Show",
        "risk_score": risk_score
    }
