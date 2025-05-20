from fastapi import FastAPI, HTTPException, Request
import numpy as np
import pickle
import uvicorn
from pydantic import BaseModel
from typing import List
import os
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Fall Detection API",
    description="API for detecting real vs fake falls using accelerometer data"
)

# Add CORS middleware to allow requests from Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production, e.g., ["your-app-domain"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model paths
MODEL_PATH = "fall_detection_model.pkl"
SCALER_PATH = "fall_detection_scaler.pkl"

# Load model and scaler on startup
@app.on_event("startup")
async def load_model():
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise Exception(f"Model files not found at {MODEL_PATH} or {SCALER_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully")

# Define request body model for accelerometer data
class SensorDataPoint(BaseModel):
    accel_x: int
    accel_y: int
    accel_z: int

class SensorDataRequest(BaseModel):
    data: List[SensorDataPoint]

class FallPredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence_level: str

@app.post("/predict_fall", response_model=FallPredictionResponse)
async def predict_fall(request: SensorDataRequest):
    try:
        # Convert incoming accelerometer data to numpy array
        data_array = np.array([
            [point.accel_x, point.accel_y, point.accel_z]
            for point in request.data
        ])

        # Simulate missing sensor data (ITG3200 gyro and MMA8451Q accel)
        # Option 1: Set missing sensor data to zeros
        itg3200_data = np.zeros_like(data_array)  # Zero out gyroscope data
        mma8451q_data = data_array  # Duplicate accelerometer data for MMA8451Q

        # Combine into 9-column format expected by the model
        full_data_array = np.hstack([data_array, itg3200_data, mma8451q_data])

        # Extract features (same as original code)
        features = []
        adxl345_acc = full_data_array[:, 0:3]
        itg3200_rot = full_data_array[:, 3:6]
        mma8451q_acc = full_data_array[:, 6:9]

        for sensor_data in [adxl345_acc, itg3200_rot, mma8451q_acc]:
            for axis in range(sensor_data.shape[1]):
                axis_data = sensor_data[:, axis]
                features.append(np.mean(axis_data))
                features.append(np.std(axis_data))
                features.append(np.min(axis_data))
                features.append(np.max(axis_data))
                features.append(np.median(axis_data))
                features.append(np.max(axis_data) - np.min(axis_data))
                features.append(np.percentile(axis_data, 25))
                features.append(np.percentile(axis_data, 75))
                features.append(np.sum(axis_data**2))
                zero_crossings = np.where(np.diff(np.signbit(axis_data)))[0]
                features.append(len(zero_crossings))
                if len(axis_data) > 2:
                    peaks = ((axis_data[1:-1] > axis_data[0:-2]) & 
                             (axis_data[1:-1] > axis_data[2:]))
                    features.append(np.sum(peaks))
                else:
                    features.append(0)

        for i in range(3):
            for j in range(i+1, 3):
                features.append(np.corrcoef(adxl345_acc[:, i], adxl345_acc[:, j])[0, 1])
                features.append(np.corrcoef(itg3200_rot[:, i], itg3200_rot[:, j])[0, 1])
                features.append(np.corrcoef(mma8451q_acc[:, i], mma8451q_acc[:, j])[0, 1])

        adxl_mag = np.sqrt(np.sum(adxl345_acc**2, axis=1))
        itg_mag = np.sqrt(np.sum(itg3200_rot**2, axis=1))
        mma_mag = np.sqrt(np.sum(mma8451q_acc**2, axis=1))

        for mag in [adxl_mag, itg_mag, mma_mag]:
            features.append(np.mean(mag))
            features.append(np.std(mag))
            features.append(np.max(mag))

        features.append(np.corrcoef(adxl_mag, mma_mag)[0, 1])
        features.append(np.corrcoef(adxl_mag, itg_mag)[0, 1])
        features.append(np.corrcoef(mma_mag, itg_mag)[0, 1])

        # Scale features
        features_scaled = scaler.transform([features])

        # Make prediction
        prediction_proba = model.predict_proba(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]

        # Define confidence level
        prob_value = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        if prob_value > 0.8:
            confidence = "High"
        elif prob_value > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "prediction": "Real Fall" if prediction == 1 else "Fake Fall",
            "probability": float(prob_value),
            "confidence_level": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": "model" in globals()}

# Debug endpoint
@app.post("/debug_data")
async def debug_data(request: Request):
    try:
        raw_data = await request.json()
        print(f"Received data: {raw_data}")
        return {"status": "Data received and logged"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)