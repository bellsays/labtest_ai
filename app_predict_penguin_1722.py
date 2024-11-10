import streamlit as st
import pandas as pd
import pickle

# Load the model and encoders
with open('model_penguin_1722.pkl', 'rb') as file:
    model_log, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Title for the Streamlit app
st.title('Penguin Species Prediction')

# Input fields for the user
st.header('Enter Penguin Details')

# Inputs for the penguin's attributes
island = st.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input('Culmen Length (mm)', min_value=0.0, max_value=100.0, step=0.1)
culmen_depth_mm = st.number_input('Culmen Depth (mm)', min_value=0.0, max_value=100.0, step=0.1)
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0, max_value=300.0, step=0.1)
body_mass_g = st.number_input('Body Mass (g)', min_value=0, max_value=10000, step=1)
sex = st.selectbox('Sex', ['MALE', 'FEMALE'])

# Create DataFrame from inputs
x_new = pd.DataFrame({
    'island': [island],  
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Encode categorical variables using the stored encoders
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# Prediction using the loaded model
if st.button('Predict'):
    prediction = model_log.predict(x_new)
    predicted_species = species_encoder.inverse_transform(prediction)

    st.subheader(f'Predicted Penguin Species: {predicted_species[0]}')

