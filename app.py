import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import base64

# --- Page Config ---
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="wide",
)

# --- Ngrok Support for Colab ---
# If you are running on Google Colab, uncomment the lines below and add your Authtoken
# from pyngrok import ngrok
# public_url = ngrok.connect(8501)
# st.sidebar.success(f"Public URL: {public_url}")

# --- Custom CSS for Premium Design ---
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #f8f9fa, #e9ecef);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #495057;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Custom Card */
    .custom-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #0b57d0 !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0b57d0 0%, #00C9FF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: transform 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* Plot margins */
    .plot-container {
        padding: 1rem;
        background: white;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- Data Engine ---
@st.cache_data
def load_data():
    try:
        # User defined archive (3).zip, but we have insurance.csv
        df = pd.read_csv("insurance.csv")
        return df
    except:
        st.error("Error: 'insurance.csv' not found. Please ensure it's in the project directory.")
        return None

@st.cache_data
def preprocess_data(df):
    df_clean = df.copy().dropna()
    
    # Encoders
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    df_clean['sex'] = le_sex.fit_transform(df_clean['sex'])
    df_clean['smoker'] = le_smoker.fit_transform(df_clean['smoker'])
    df_clean['region'] = le_region.fit_transform(df_clean['region'])
    
    # Scaler for BMI
    scaler = StandardScaler()
    df_clean['bmi_scaled'] = scaler.fit_transform(df_clean[['bmi']])
    
    return df_clean, le_sex, le_smoker, le_region, scaler

@st.cache_resource
def train_models(df):
    X = df[["age", "bmi_scaled", "smoker", "region"]]
    y = df["charges"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)
    
    return lr, rf, gb, X_train, X_test, y_train, y_test

# --- Load Main Data ---
raw_df = load_data()

if raw_df is not None:
    df, le_sex, le_smoker, le_region, scaler = preprocess_data(raw_df)
    lr, rf, gb, X_train, X_test, y_train, y_test = train_models(df)

    # --- UI Layout ---
    st.markdown('<h1 class="main-header">Insurance AI Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced health insurance cost forecasting using machine learning.</p>', unsafe_allow_html=True)

    # --- Sidebar Inputs ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=100)
        st.header("Patient Profile")
        
        age = st.slider("Age", 18, 100, 30)
        bmi = st.number_input("BMI Index", 10.0, 60.0, 25.0, help="Body Mass Index")
        sex_input = st.selectbox("Gender", options=le_sex.classes_)
        smoker_input = st.selectbox("Smoking Status", options=le_smoker.classes_)
        region_input = st.selectbox("Region", options=le_region.classes_)
        
        st.markdown("---")
        mode_select = st.selectbox("Insight Mode", ["Prediction", "Data Analysis", "Model Comparison"])

    # --- Main Content Area ---
    if mode_select == "Prediction":
        col_inp, col_res = st.columns([1, 1])
        
        with col_inp:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("Selected Profile")
            st.info(f"**Age:** {age} years\n\n**BMI:** {bmi}\n\n**Sex:** {sex_input}\n\n**Smoker:** {smoker_input}\n\n**Region:** {region_input}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("Price Forecast")
            
            # Prepare data for prediction
            smoker_enc = le_smoker.transform([smoker_input])[0]
            region_enc = le_region.transform([region_input])[0]
            bmi_scaled = scaler.transform([[bmi]])[0][0]
            
            X_input = [[age, bmi_scaled, smoker_enc, region_enc]]
            
            # Using Random Forest as default for forecast
            rf_pred = rf.predict(X_input)[0]
            gb_pred = gb.predict(X_input)[0]
            lr_pred = lr.predict(X_input)[0]
            
            avg_pred = (rf_pred + gb_pred + lr_pred) / 3
            
            st.metric("Estimated Charges", f"${avg_pred:,.2f}", delta=f"{avg_pred - 13270:,.2f} vs Avg")
            st.caption("Average prediction from 3 distinct ML architectures.")
            
            # Show individual model outputs in a small table
            model_outputs = pd.DataFrame({
                "Architecture": ["Random Forest", "Gradient Boosting", "Linear Regression"],
                "Forecast": [f"${rf_pred:,.2f}", f"${gb_pred:,.2f}", f"${lr_pred:,.2f}"]
            })
            st.table(model_outputs)
            st.markdown('</div>', unsafe_allow_html=True)

    elif mode_select == "Data Analysis":
        st.subheader("Exploratory Data Insights")
        t1, t2, t3 = st.tabs(["Cost Distribution", "Correlations", "Impact Analysis"])
        
        with t1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Charges Frequency**")
                fig_hist = plt.figure(figsize=(10,6))
                sns.histplot(raw_df["charges"], kde=True, color="#00C9FF")
                plt.title("Distribution of Insurance Charges")
                st.pyplot(fig_hist)
            with col_b:
                st.write("**Smoking Impact**")
                fig_box = plt.figure(figsize=(10,6))
                sns.boxplot(x="smoker", y="charges", data=raw_df, palette="viridis")
                plt.title("Smoking vs Insurance Charges")
                st.pyplot(fig_box)

        with t2:
            st.write("**Feature Relationships**")
            # Numeric only correlate
            df_numeric = df.select_dtypes(include=[np.number])
            fig_corr = plt.figure(figsize=(12,8))
            sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Feature Correlation Matrix")
            st.pyplot(fig_corr)

        with t3:
            st.write("**Predictor Importance (Random Forest)**")
            importance = rf.feature_importances_
            feat_names = ["age", "bmi", "smoker", "region"]
            fig_imp = plt.figure(figsize=(10,6))
            sns.barplot(x=importance, y=feat_names, palette="magma")
            plt.title("Relative Impact of Features on Insurance Costs")
            st.pyplot(fig_imp)

    elif mode_select == "Model Comparison":
        st.subheader("Machine Learning Performance Evaluation")
        
        lr_p = lr.predict(X_test)
        rf_p = rf.predict(X_test)
        gb_p = gb.predict(X_test)
        
        results = pd.DataFrame({
            "Metric": ["RMSE (Lower is better)", "MAE (Lower is better)"],
            "Random Forest": [np.sqrt(mean_squared_error(y_test, rf_p)), mean_absolute_error(y_test, rf_p)],
            "Gradient Boosting": [np.sqrt(mean_squared_error(y_test, gb_p)), mean_absolute_error(y_test, gb_p)],
            "Linear Regression": [np.sqrt(mean_squared_error(y_test, lr_p)), mean_absolute_error(y_test, lr_p)]
        })
        
        st.dataframe(results.style.format(precision=2).highlight_min(axis=1, color='lightgreen'))
        
        st.info("💡 **Gradient Boosting** and **Random Forest** typically outperform Linear Regression on this dataset due to their ability to capture non-linear relationships (like the interaction between age and smoking).")

    # --- Footer ---
    st.markdown("---")
    st.markdown("Created with ❤️ for Insurance Project | AI Analytics Dashboard")
