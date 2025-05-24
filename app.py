import streamlit as st
import pandas as pd
import joblib


# Stylish background and font using CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Quicksand', sans-serif;
        background-color: #E3F2FD;
        color: #2E3A59;
    }

    .heading-box {
        background: linear-gradient(90deg, #D1C4E9, #B39DDB);
        color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        margin-bottom: 30px;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)
# Show heading in the box
st.markdown('<div class="heading-box">🚗 Car Price Prediction App</div>', unsafe_allow_html=True)

# Load your saved model
model = joblib.load('best_car_price_model.pkl')

# Input widgets for all features
car_name = st.selectbox('Car Name :', ['CarA', 'CarB', 'CarC'])  # replace with your actual car names
year = st.number_input('Year of Purchase :', min_value=1990, max_value=2025, value=2015)
present_price = st.number_input('Present Price (in lakhs) :', min_value=0.0, max_value=100.0, step=0.01)
kms_driven = st.number_input('Kilometers Driven :', min_value=0, max_value=1000000, step=100)
fuel_type = st.selectbox('Fuel Type :', ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox('Seller Type :', ['Dealer', 'Individual'])
transmission = st.selectbox('Transmission :', ['Manual', 'Automatic'])
owner = st.selectbox('Owner :', [0, 1, 2, 3])

# On button click, make prediction
if st.button('Predict Selling Price'):
    input_df = pd.DataFrame({
        'Car_Name': [car_name],
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })

    prediction = model.predict(input_df)
    st.success(f"Predicted Selling Price: ₹{prediction[0]:.2f} Lakhs")

    st.toast("Prediction successful!", icon="🎉")

