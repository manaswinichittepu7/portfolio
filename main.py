from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("final_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define the input schema with exact feature names
class LungCancerPredictionRequest(BaseModel):
    GENDER: int
    AGE: int
    SMOKING: int
    YELLOW_FINGERS: int
    ANXIETY: int
    PEER_PRESSURE: int
    CHRONIC_DISEASE: int
    FATIGUE: int
    ALLERGY: int
    WHEEZING: int
    ALCOHOL_CONSUMING: int
    COUGHING: int
    SHORTNESS_OF_BREATH: int
    SWALLOWING_DIFFICULTY: int
    CHEST_PAIN: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Lung Cancer Prediction API"}

@app.post("/predict/")
def predict(input_data: LungCancerPredictionRequest):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Ensure the DataFrame columns match the model's expected feature order
    input_df = input_df[model.feature_names_in_]

    # Make the prediction
    prediction = model.predict(input_df)

    return {"prediction": int(prediction[0])}
