import streamlit as st
import numpy as np
import pickle
import pandas as pd

Sc , model = pickle.load(open('diabetes_model2.pkl','rb'))

st.title("Diabetes Predicion App")
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 10, 100)

if st.button("Predict"):
    data = {
        "Pregnancies":pregnancies,
        "Glucose":glucose,
        "BloodPressure":bp,
        "SkinThickness":skin_thickness,
        "Insulin":insulin,
        "BMI":bmi,
        "DiabetesPedigreeFunction":dpf,
        "Age":age
    }

    new_Data = pd.DataFrame([data])
    Transformed_Data = Sc.transform(new_Data)
    result  = model.predict(Transformed_Data)

    st.success("Diabetic" if result[0] == 1 else "Not Diabetic")




