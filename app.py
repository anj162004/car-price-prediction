import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import numpy as np
import os
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from streamlit.components.v1 import components
# ========== Theme Settings ==========
bg_color = "#FFFFFF"
font_color = "#000000"

# ========== Custom CSS ==========
st.markdown(f"""
    <style>
    button[data-baseweb="tab"] span {{
        color: #000 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ========== Global Styling ==========
# Fix label visibility in dark mode
st.markdown("""
<style>
label[data-testid="stCheckboxLabel"] {
    color: #000 !important;      /* Use #fff for white if using dark background */
    font-weight: 600 !important;
    font-size: 16px !important;
    display: block;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div.row-widget.stCheckbox {
    padding: 6px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
     <style>
    html, body, [class*="css"] {
        background-color: white !important;
        /* REMOVE this: color: black !important; */
    }
    .form-label {
        font-weight: bold;
        color: #333333 !important;  /* Now this will work */
        margin-bottom: 4px;
        display: block;
    }
    button[data-testid="predict_button"] {
        color: white !important;
        background-color: black !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 16px;
    }
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&display=swap');
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'Quicksand', sans-serif;
        animation: fadeIn 0.8s ease-in-out;
    }
    .heading-box {
        background: linear-gradient(to right, #8e9eab, #eef2f3);
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
    .result-box {
        animation: slideFade 0.6s ease;
        background-color: #f4f9ff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideFade {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# ========== SHAP Utility ==========
def st_shap(plot, height=None):
    if hasattr(plot, "html"):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height or 500, scrolling=True)
    else:
        st.warning("This SHAP plot cannot be rendered with st_shap(). Use st.pyplot() instead.")

# ========== Preprocessor ==========
numeric_features = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Car_Name']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ========== Caching ==========
@st.cache_resource
def load_model():
    return joblib.load('best_car_price_model.pkl')

@st.cache_resource
def load_data():
    return pd.read_csv('car data.csv')

# ========== SHAP Explainer ==========
def get_shap_explainer(model, df, drop_target='Selling_Price'):
    preprocessor = model.named_steps['preprocessor']
    final_model = model.named_steps['model']
    shap_df = df.drop(drop_target, axis=1, errors="ignore")
    X_transformed = preprocessor.transform(shap_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    X_df_named = pd.DataFrame(X_transformed, columns=feature_names)
    explainer = shap.Explainer(final_model, X_df_named)
    return explainer, X_df_named

# ========== Load ==========
model = load_model()
df = load_data()
explainer, X_df_named = get_shap_explainer(model, df)

# ========== App Header ==========
st.markdown('<div class="heading-box">🚗 Car Price Prediction App</div>', unsafe_allow_html=True)
tab3, tab1, tab2 = st.tabs(["ℹ️ About the App", "🚘 Predict Price", "📊 Dashboard"])

# ========== Tab: About ==========
with tab3:
    st.markdown("<h3 style='color:black !important;'>ℹ️ About This App</h3>", unsafe_allow_html=True)
    st.markdown("""
    This app predicts the selling price of used cars using a trained ML model.
    - Input car details
    - Get price predictions
    - Explore SHAP visualizations and insights
    """)
    banner_path = "car_banner.gif"
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
    else:
        st.warning(f"Banner image not found at: {banner_path}")

# ========== Tab: Predict Price ==========
with tab1:
    st.markdown("<h3 style='color:black; font-weight:bold;'>📝 Car Details Form</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='
        background: linear-gradient(to right, #fceabb, #f8b500);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 25px;
    '>
""", unsafe_allow_html=True)

    with st.form("car_form"):

        st.markdown('<span style="font-weight:bold; color:#1f77b4;">📅 Year</span>', unsafe_allow_html=True)
        year = st.number_input(" ", min_value=1990, max_value=2025, value=2015)

        st.markdown('<span style="font-weight:bold; color:#e67e22;">💰 Present Price (in Lakhs)</span>', unsafe_allow_html=True)
        present_price = st.number_input("  ", min_value=0.0, value=5.0)

        st.markdown('<span style="font-weight:bold; color:#27ae60;">🛣️ Kms Driven</span>', unsafe_allow_html=True)
        kms = st.number_input("   ", min_value=0, value=50000)

        st.markdown('<span style="font-weight:bold; color:#9b59b6;">👤 Owner Count</span>', unsafe_allow_html=True)
        owner = st.selectbox("    ", ["0", "1", "2", "3"])

        st.markdown('<span style="font-weight:bold; color:#e74c3c;">⛽ Fuel Type</span>', unsafe_allow_html=True)
        fuel = st.selectbox("     ", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

        st.markdown('<span style="font-weight:bold; color:#2980b9;">🏪 Seller Type</span>', unsafe_allow_html=True)
        seller = st.selectbox("      ", ["Dealer", "Individual", "Trustmark Dealer"])

        st.markdown('<span style="font-weight:bold; color:#16a085;">⚙️ Transmission</span>', unsafe_allow_html=True)
        transmission = st.selectbox("       ", ["Manual", "Automatic"])

        st.markdown('<span style="font-weight:bold; color:#d35400;">🚗 Car Name</span>', unsafe_allow_html=True)
        car_name = st.text_input("        ", "Maruti Swift")

        submitted = st.form_submit_button("Predict Price")
        st.markdown("</div>", unsafe_allow_html=True)



        if submitted:
            input_df = pd.DataFrame([{
                "Year": year,
                "Present_Price": present_price,
                "Kms_Driven": kms,
                "Owner": int(owner),
                "Fuel_Type": fuel,
                "Seller_Type": seller,
                "Transmission": transmission,
                "Car_Name": car_name
            }])
            input_df[categorical_features] = input_df[categorical_features].astype(str)

            st.write("### Input DataFrame:")
            st.dataframe(input_df)

            try:
                preprocessor = model.named_steps['preprocessor']
                final_model = model.named_steps['model']
                input_transformed = preprocessor.transform(input_df)
                if hasattr(input_transformed, "toarray"):
                    input_transformed = input_transformed.toarray()

                prediction = final_model.predict(input_transformed)[0]
                st.markdown(f"<div class='result-box'><h4>Estimated Price: ₹ {prediction:,.2f} Lakhs</h4></div>", unsafe_allow_html=True)

                single_input = pd.DataFrame(input_transformed, columns=preprocessor.get_feature_names_out())
                shap_input_values = explainer(single_input, check_additivity=False)

                st.subheader("🔍 Feature Contribution for this Prediction")
                st.markdown("""
                The **SHAP Waterfall Plot** below helps explain *why* the model predicted the price it did:
                - 📈 Features increasing price are red
                - 📉 Features decreasing price are blue
                - 🚗 Top shows final predicted price
                """)
                shap.plots.waterfall(shap_input_values[0], max_display=10, show=False)
                st.pyplot(plt.gcf())

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ========== Tab: Dashboard ==========
with tab2:
    st.markdown("<h2 style='color:black !important;'>📊 Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Explore the dataset used to train the model.")
    st.markdown("""
<div style='background-color: #f9f9f9; padding: 12px 20px; border-left: 5px solid #4a90e2; border-radius: 6px;'>
<h4 style='color:black;'>📊 Feature Visualization</h4>
<p style='color:black; font-size: 15px;'>
Choose a feature from the dropdown to see its distribution:
<ul>
<li><b>Categorical Features</b> (like Fuel Type or Seller Type) show frequency using a bar chart.</li>
<li><b>Numerical Features</b> (like Year or Present Price) show distribution using a histogram and smooth density curve.</li>
</ul>
This helps you understand data trends and variability in your dataset.
</p>
</div>
""", unsafe_allow_html=True)


    selected_col = st.selectbox("Select a feature to visualize", df.columns)

    if df[selected_col].dtype == 'object':
        fig = plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=selected_col)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        fig = plt.figure(figsize=(10, 5))
        sns.histplot(df[selected_col], kde=True, color='skyblue')
        st.pyplot(fig)

    # Style checkbox labels globally for better mobile visibility
    # Force checkbox label to be bold and visible across all devices
    st.markdown("""
    <style>
    /* Force all checkbox labels to be visible and bold */
    [data-testid="stCheckbox"] label, label[data-testid="stCheckboxLabel"] {
        font-weight: 800 !important;
        font-size: 18px !important;
        color: #000000 !important;
        display: block !important;
        line-height: 1.8 !important;
        white-space: normal !important;
    }

    /* Make sure container spacing is clean on mobile */
    @media screen and (max-width: 768px) {
        [data-testid="stCheckbox"] {
            margin-bottom: 12px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)



    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("📌 Feature Correlation")
        corr = df.select_dtypes(include=np.number).corr()
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig)
        st.markdown("""
<div style='background-color: #f9f9f9; padding: 12px 20px; border-left: 5px solid #2a9df4; border-radius: 6px;'>
<h4 style='color:black;'>📌 What is a Feature Correlation Heatmap?</h4>
<p style='color:black; font-size: 15px;'>
This heatmap shows the strength of relationships between numerical features using correlation values:
<ul>
<li><b>+1</b>: Perfect positive correlation – both features increase together.</li>
<li><b>-1</b>: Perfect negative correlation – one increases as the other decreases.</li>
<li><b>0</b>: No correlation.</li>
</ul>
This helps identify related features and potential redundancies.
</p>
</div>
""", unsafe_allow_html=True)


    if st.checkbox("Show SHAP Summary Plot"):
        st.subheader("🔍 SHAP Summary Plot (Global Feature Importance)")
        shap_values_all = explainer(X_df_named, check_additivity=False)
        plt.figure()
        shap.summary_plot(shap_values_all.values, X_df_named)
        st.pyplot(plt.gcf())
        st.markdown("""
<div style='background-color: #f9f9f9; padding: 12px 20px; border-left: 5px solid #f77b00; border-radius: 6px;'>
<h4 style='color:black;'>🔍 What is a SHAP Summary Plot?</h4>
<p style='color:black; font-size: 15px;'>
This plot explains the model's predictions across the dataset:
<ul>
<li><b>Y-axis:</b> Features ranked by overall importance.</li>
<li><b>Color:</b> Red = high feature value, Blue = low value.</li>
<li><b>X-axis:</b> Direction of impact – right increases prediction, left decreases it.</li>
</ul>
It gives a global view of how each feature contributes to model decisions.
</p>
</div>
""", unsafe_allow_html=True)
