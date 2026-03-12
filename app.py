import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Hospital Readmission Risk Prediction")

age = st.number_input("Age",20,100)
previous_admissions = st.number_input("Previous Admissions",0,10)
length_of_stay = st.number_input("Length of Stay (Days)",1,30)
diabetes = st.selectbox("Diabetes",[0,1])
hypertension = st.selectbox("Hypertension",[0,1])
heart_disease = st.selectbox("Heart Disease",[0,1])
medications = st.number_input("Number of Medications",0,20)

if st.button("Predict"):

    features = np.array([[age,previous_admissions,length_of_stay,
                          diabetes,hypertension,heart_disease,medications]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("High Risk of Readmission")
    else:
        st.success("Low Risk of Readmission")