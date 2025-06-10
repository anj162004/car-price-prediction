import streamlit as st
from streamlit_toggle import st_toggle_switch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import shap
st.set_page_config(page_title="Car Price App", page_icon="🚗")


# ========== Theme Toggle ==========
theme_toggle = st_toggle_switch('🌓 Toggle Theme', label_after=True, default_value=False)

# ========== Apply Theme Based on Toggle ==========
if theme_toggle:
    bg_color = "#121212"
    font_color = "#f1f1f1"
    gradient = "linear-gradient(90deg, #333333, #666666)"
else:
    bg_color = "#E3F2FD"
    font_color = "#2E3A59"
    gradient = "linear-gradient(90deg, #D1C4E9, #B39DDB)"

# ========== Inject CSS and JS ==========
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
    .theme-float-box {{
        position: fixed;
        top: 10px;
        right: 20px;
        z-index: 9999;
        background-color: white;
        padding: 5px 12px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        font-size: 14px;
        width: 160px;
    }}
    </style>
    <div class="theme-float-box" id="theme-container"></div>
    <script>
    const container = window.parent.document.querySelector("#theme-container");
    const widget = window.parent.document.querySelectorAll('[data-baseweb="select"]')[0];
    if (container && widget && !container.contains(widget)) {{
        container.appendChild(widget);
    }}
    </script>
""", unsafe_allow_html=True)

# ========== Load Model and Data ==========
model = joblib.load('best_car_price_model.pkl')
df = pd.read_csv('car data.csv')

# ========== Heading ==========
st.markdown('<div class="heading-box">🚗 Car Price Prediction App</div>', unsafe_allow_html=True)

# ========== Tabs ==========
tab1, tab2 = st.tabs(["🚘 Predict Price", "📊 Dashboard"])

# ========== Tab 1 ==========
with tab1:
    st.subheader("Enter car details")

    car_name = st.selectbox('🚘 Car Name :', ['CarA', 'CarB', 'CarC'])
    year = st.number_input('📅 Year of Purchase :', min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input('💰 Present Price (in lakhs) :', min_value=0.0, max_value=100.0, step=0.01)
    kms_driven = st.number_input('🛣️ Kilometers Driven :', min_value=0, max_value=1000000, step=100)
    fuel_type = st.selectbox('⛽ Fuel Type :', ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox('🧍 Seller Type :', ['Dealer', 'Individual'])
    transmission = st.selectbox('⚙️ Transmission :', ['Manual', 'Automatic'])
    owner = st.selectbox('👥 Owner Count :', [0, 1, 2, 3])

    st.sidebar.markdown("### ✅ Your Input Summary")
    st.sidebar.write(f"**Car Name:** {car_name}")
    st.sidebar.write(f"**Year of Purchase:** {year}")
    st.sidebar.write(f"**Present Price (in Lakhs):** {present_price}")
    st.sidebar.write(f"**Kilometers Driven:** {kms_driven}")
    st.sidebar.write(f"**Fuel Type:** {fuel_type}")
    st.sidebar.write(f"**Seller Type:** {seller_type}")
    st.sidebar.write(f"**Transmission:** {transmission}")
    st.sidebar.write(f"**Owner Count:** {owner}")

    if st.button('🚀 Predict Selling Price'):
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

        with st.spinner("Predicting..."):
            time.sleep(2)
            prediction = model.predict(input_df)
            st.success(f"Predicted Selling Price: ₹{prediction[0]:.2f} Lakhs")
            st.toast("Prediction successful!", icon="🎉")

        # SHAP Explanation
        st.subheader("🔍 Why this Prediction?")
        preprocessor = model.named_steps['preprocessor']
        final_model = model.named_steps['model']

        shap_df = df.drop("Selling Price", axis=1, errors="ignore")
        X_transformed = preprocessor.transform(shap_df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        input_transformed = preprocessor.transform(input_df)
        if hasattr(input_transformed, "toarray"):
            input_transformed = input_transformed.toarray()

        # Clean feature names
        feature_names = [name.split('__')[-1].replace('_', ' ') for name in preprocessor.get_feature_names_out()]
        X_df_named = pd.DataFrame(X_transformed, columns=feature_names)
        input_df_named = pd.DataFrame(input_transformed, columns=feature_names)

        explainer = shap.Explainer(final_model, X_df_named)
        shap_values = explainer(input_df_named)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

# ========== Tab 2 ==========
with tab2:
    st.header("📈 Car Dataset Dashboard")
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head())

    st.subheader("💰 Present Price Distribution")
    st.bar_chart(df['Present_Price'])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Fuel Type Distribution")
        st.bar_chart(df['Fuel_Type'].value_counts())
    with col2:
        st.subheader("🧍 Seller Type Breakdown")
        st.bar_chart(df['Seller_Type'].value_counts())

    st.subheader("📈 Yearly Price Distribution")
    yearly_df = df.groupby('Year')["Present_Price"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data=yearly_df, x="Year", y="Present_Price", ax=ax, marker="o", color="purple")
    ax.set_ylabel("Average Price (in Lakhs) →")
    st.pyplot(fig)

    st.subheader("📉 Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # SHAP Summary Plot
    st.subheader("🧠 SHAP Summary Plot (Feature Impact Overview)")
    preprocessor = model.named_steps['preprocessor']
    final_model = model.named_steps['model']

    shap_df = df.drop("Selling Price", axis=1, errors="ignore")


    X_transformed = preprocessor.transform(shap_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = [name.split('__')[-1].replace('_', ' ') for name in preprocessor.get_feature_names_out()]
    X_df_named = pd.DataFrame(X_transformed, columns=feature_names)

    explainer = shap.Explainer(final_model, X_df_named)
    shap_values = explainer(X_df_named)

    fig, ax = plt.subplots(figsize=(12, 6))
    shap.summary_plot(shap_values, X_df_named, plot_type="bar", show=False)
    st.pyplot(fig)
