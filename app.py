import streamlit as st
import requests

# Streamlit app title
st.title("Lung Cancer Prediction App")
st.write("Interact with the deployed API to predict lung cancer likelihood.")

# Input fields for user data
gender = st.selectbox("Gender (1 = Male, 0 = Female)", options=[1, 0])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
smoking = st.selectbox("Smoking (1 = Yes, 0 = No)", options=[1, 0])
yellow_fingers = st.selectbox("Yellow Fingers (1 = Yes, 0 = No)", options=[1, 0])
anxiety = st.selectbox("Anxiety (1 = Yes, 0 = No)", options=[1, 0])
peer_pressure = st.selectbox("Peer Pressure (1 = Yes, 0 = No)", options=[1, 0])
chronic_disease = st.selectbox("Chronic Disease (1 = Yes, 0 = No)", options=[1, 0])
fatigue = st.selectbox("Fatigue (1 = Yes, 0 = No)", options=[1, 0])
allergy = st.selectbox("Allergy (1 = Yes, 0 = No)", options=[1, 0])
wheezing = st.selectbox("Wheezing (1 = Yes, 0 = No)", options=[1, 0])
alcohol_consuming = st.selectbox("Alcohol Consuming (1 = Yes, 0 = No)", options=[1, 0])
coughing = st.selectbox("Coughing (1 = Yes, 0 = No)", options=[1, 0])
shortness_of_breath = st.selectbox("Shortness of Breath (1 = Yes, 0 = No)", options=[1, 0])
swallowing_difficulty = st.selectbox("Swallowing Difficulty (1 = Yes, 0 = No)", options=[1, 0])
chest_pain = st.selectbox("Chest Pain (1 = Yes, 0 = No)", options=[1, 0])

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = {
        "GENDER": gender,
        "AGE": age,
        "SMOKING": smoking,
        "YELLOW_FINGERS": yellow_fingers,
        "ANXIETY": anxiety,
        "PEER_PRESSURE": peer_pressure,
        "CHRONIC_DISEASE": chronic_disease,
        "FATIGUE": fatigue,
        "ALLERGY": allergy,
        "WHEEZING": wheezing,
        "ALCOHOL_CONSUMING": alcohol_consuming,
        "COUGHING": coughing,
        "SHORTNESS_OF_BREATH": shortness_of_breath,
        "SWALLOWING_DIFFICULTY": swallowing_difficulty,
        "CHEST_PAIN": chest_pain,
    }

    # API endpoint URL
    api_url = "http://127.0.0.1:8000/predict/"
    response = requests.post(api_url, json=input_data)

    # Display the prediction result
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Lung Cancer Prediction: {'Positive' if prediction == 1 else 'Negative'}")
    else:
        st.error(f"Error: {response.status_code}. Response: {response.text}")
