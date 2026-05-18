from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Input schema with all features
class InputData(BaseModel):
    holiday: int
    workingday: int
    weathersit_Clear: int
    weathersit_Mist: int
    weathersit_Light_Snow: int
    weathersit_Heavy_Rain: int
    season: int
    hr: int
    weekday: int
    temp: float
    atemp: float
    hum: float
    windspeed: float
    day: int
    month: int
    year: int

# Load trained model
model = pickle.load(open("Lgbmodel.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Bike Demand Prediction API"}

@app.post("/predict")
def predict(data: InputData):

    input_dict = data.dict()

    df = pd.DataFrame([input_dict])

    prediction = model.predict(df)

    return {
        "predicted_bike_rentals": int(prediction[0])
    }