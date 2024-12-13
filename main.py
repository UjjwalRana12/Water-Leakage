from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os


class InputData(BaseModel):
    temp: float
    waterflow: float
    pressure: float


app = FastAPI()


model = load_model('aquaAlert.h5')
scaler_path = 'scaler.pkl'

if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
scaler = joblib.load(scaler_path)

@app.get("/")
async def get():
    return {"message": "use /docs to see the documentation"}


@app.post("/predict")
async def predict(data: InputData):
   
    new_data = np.array([[data.temp, data.waterflow, data.pressure]])
    
    
    scaled_data = scaler.transform(new_data)
    reshaped_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))
    
  
    predictions = model.predict(reshaped_data)
    
    
    threshold = 0.5
    predicted_class = int((predictions > threshold).astype(int)[0][0])
    
    return {
        "temp": data.temp,
        "waterflow": data.waterflow,
        "pressure": data.pressure,
        "predicted_probability": float(predictions[0][0]),
        "predicted_class": predicted_class
    }
