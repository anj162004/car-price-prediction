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

def st_shap(plot, height=None):
    """Use only for JavaScript-based SHAP plots like force_plot"""
    if hasattr(plot, "html"):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height or 500, scrolling=True)
    else:
        st.warning("This SHAP plot cannot be rendered with st_shap(). Use st.pyplot() instead.")

# ========== Theme Settings ==========
bg_color = "#FFFFFF"
font_color = "#000000"
gradient = "linear-gradient(to right, #8e9eab, #eef2f3)"

# ========== Preprocessor Features ==========
numeric_features = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Car_Name']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# ========== Custom CSS ==========
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&display=swap');
    body {{
        background-color: {bg_color} !important;
    }}
    .stApp {{
        background-color: {bg_color} !important;
        color: {font_color} !important;
        font-family: 'Quicksand', sans-serif;
        animation: fadeIn 0.8s ease-in-out;
    }}
    .heading-box {{
        background: {gradient};
        color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        margin-bottom: 30px;
        letter-spacing: 1px;
    }}
    .result-box {{
        animation: slideFade 0.6s ease;
        background-color: #f4f9ff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }}
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(10px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes slideFade {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
""", unsafe_allow_html=True)




# ========== Load Cached Resources ==========
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

# ========== Title ==========
st.markdown('<div class="heading-box">🚗 Car Price Prediction App</div>', unsafe_allow_html=True)

# ========== Tabs ==========
tab3, tab1, tab2 = st.tabs(["ℹ️ About the App", "🚘 Predict Price", "📊 Dashboard"])

# ========== Tab 3 ==========
with tab3:
    st.header("ℹ️ About This App")
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

# ========== Tab 1: Prediction ==========
with tab1:
    with st.form("input_form"):
        year = st.number_input("📅 Year", min_value=1990, max_value=2025, value=2015)
        present_price = st.number_input("💰 Present Price (in Lakhs)", min_value=0.0, value=5.0)
        kms = st.number_input("🛣️ Kms Driven", min_value=0, value=50000)
        owner = st.selectbox("👤 Owner Count", ["0", "1", "2", "3"])
        fuel = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        seller = st.selectbox("🏪 Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
        transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])
        car_name = st.text_input("🚗 Car Name", "Maruti Swift")
        submitted = st.form_submit_button("🧮 Predict Price")

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

            # SHAP Waterfall Plot
            single_input = pd.DataFrame(input_transformed, columns=preprocessor.get_feature_names_out())
            shap_input_values = explainer(single_input, check_additivity=False)

            st.subheader("🔍 Feature Contribution for this Prediction")
            st.markdown("""
The **SHAP Waterfall Plot** below helps explain *why* the model predicted the price it did, by breaking down the impact of each feature:
- 💡 **Base Value** (bottom): The average predicted price across all cars in the training set.
- 📈 **Positive Contributions** (in red): Features that **increase** the predicted price.
- 📉 **Negative Contributions** (in blue): Features that **decrease** the predicted price.
- 🚗 The top of the plot shows the **final predicted price** after adding all effects.

This lets you see how much each input feature (like Present Price, Fuel Type, Year, etc.) influenced the final prediction.
""")

            shap.plots.waterfall(shap_input_values[0], max_display=10, show=False)
            st.pyplot(plt.gcf())

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ========== Tab 2: Dashboard ==========
with tab2:
    st.header("📊 Dashboard")
    st.markdown("Explore the dataset used to train the model.")
    

    selected_col = st.selectbox("Select a feature to visualize", df.columns)
    st.markdown(f"""
        🔠 **{selected_col} Count Plot**  
        This bar chart shows how many times each value appears in the data.  
        For example, how many cars use petrol or how many sellers are individuals.
        """)
    if df[selected_col].dtype == 'object':
        st.markdown(f"""
        📊 **{selected_col} Distribution**  
        This chart shows how the values are spread out.  
        For example, are most cars priced low or high? It helps spot patterns and outliers.
        """)
        fig = plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=selected_col)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        fig = plt.figure(figsize=(10, 5))
        sns.histplot(df[selected_col], kde=True, color='skyblue')
        st.pyplot(fig)

    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Feature Correlation")
        st.markdown("""
        🔗 **Correlation Heatmap**  
        This heatmap shows how related two features are.  
        A value close to 1 means strong relation, like how 'Year' and 'Price' might be connected.
        """)
        corr = df.select_dtypes(include=np.number).corr()
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig)

    if st.checkbox("Show SHAP Summary Plot"):
        st.subheader("🔍 SHAP Summary Plot (Global Feature Importance)")
        st.markdown("""
    This plot shows how each feature in the dataset affects the model's predictions:
    - 🎯 **Features on the Y-axis** are sorted by importance (top = most impactful).
    - 🎨 **Color** represents the value of the feature (red = high, blue = low).
    - ➕ A feature that pushes predictions higher appears more on the **right**.
    - ➖ A feature that lowers predictions appears more on the **left**.
    """)
    
        shap_values_all = explainer(X_df_named, check_additivity=False)

        import matplotlib.pyplot as plt
        plt.figure()
        shap.summary_plot(shap_values_all.values, X_df_named)
        st.pyplot(plt.gcf())

