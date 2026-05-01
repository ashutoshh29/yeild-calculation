from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("crop_yield_model.pkl")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame({
        "Crop": [data["crop"]],
        "Crop_Year": [data["year"]],
        "Season": [data["season"]],
        "State": [data["state"]],
        "Area": [data["area"]],
        "Annual_Rainfall": [data["rainfall"]],
        "Fertilizer": [data["fertilizer"]],
        "Pesticide": [data["pesticide"]]
    })

    prediction = model.predict(input_data)

    return {"predicted_yield": float(prediction[0])}