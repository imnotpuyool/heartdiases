from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API untuk memprediksi risiko penyakit jantung menggunakan SVM",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


class InputData(BaseModel):
    features: list


@app.get("/")
def home():
    return {"status": "Heart Disease SVM API Running"}


@app.get("/metrics")
def get_metrics():
    return {
        "model_name": "Support Vector Machine (SVM)",
        "kernel": "RBF (C=5.0)",
        "test_accuracy": "97.07%",
        "precision": "97%",
        "recall": "97%",
        "dataset_rows": 1025,
        "status": "Optimal"
    }


@app.post("/predict")
def predict(data: InputData):
    x = np.array(data.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prediction = int(model.predict(x_scaled)[0])
    probabilities = model.predict_proba(x_scaled)[0]
    prob_healthy = float(probabilities[0])
    prob_disease = float(probabilities[1])
    probability = float(probabilities.max())
    label = "Terindikasi Penyakit Jantung" if prediction == 1 else "Normal / Tidak Terindikasi Penyakit Jantung"

    return {
        "prediction": prediction,
        "label": label,
        "probability": probability,
        "prob_disease": prob_disease,
        "prob_healthy": prob_healthy
    }