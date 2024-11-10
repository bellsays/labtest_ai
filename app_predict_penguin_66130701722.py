

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# โหลดโมเดลและ encoders ที่จำเป็น
with open('model_penguin_66130701722.pkl', 'rb') as file:
    model_log, species_encoder, island_encoder, sex_encoder = pickle.load(file)

st.title("Penguin Species Predictor")

# รับข้อมูลจากผู้ใช้
island = st.selectbox("Select Island", options=['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=0.0)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=0.0)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0)
body_mass_g = st.number_input("Body Mass (g)", min_value=0.0)
sex = st.selectbox("Sex", options=['MALE', 'FEMALE'])

# เมื่อกดปุ่ม "Predict"
if st.button("Predict"):
    # เตรียมข้อมูลจากอินพุตของผู้ใช้
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]
    })

    # ใช้ encoders แปลงข้อมูล
    try:
        x_new['island'] = island_encoder.transform(x_new['island'])
        x_new['sex'] = sex_encoder.transform(x_new['sex'])
    except Exception as e:
        st.error(f"Error during encoding: {e}")

    # ตรวจสอบค่าว่าไม่มี NaN และแปลงคอลัมน์ให้เป็นตัวเลข
x_new = x_new.apply(pd.to_numeric, errors='coerce')
if x_new.isnull().values.any():
    st.error("There is a NaN value in the input data. Please check the input values.")
else:
    # ทำนายผลลัพธ์
    try:
        y_pred_new = model_log.predict(x_new)
        result = species_encoder.inverse_transform(y_pred_new)
        st.success(f'Predicted Species: {result[0]}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
