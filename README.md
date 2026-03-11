<<<<<<< HEAD
# Insurance-Charges-Predictor
=======
# Insurance Charges Predictor 🏥

An advanced health insurance cost forecasting dashboard built with Python, Streamlit, and Machine Learning.

## Overview
This application predicts medical insurance costs based on patient profiles using multiple machine learning architectures:
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Linear Regression**

The app also provides exploratory data analysis, visualizations of feature importance, and model performance comparisons.

## Features
- **Interactive Patient Profile:** Input age, BMI, gender, smoking status, and region to get a customized cost estimate.
- **Ensemble Prediction:** Averages predictions from 3 distinct models for higher reliability.
- **Exploratory Data Analysis:** View cost distributions, feature correlations, and the relative impact of predictors on insurance costs.
- **Model Comparison:** Evaluates and compares algorithms using RMSE and MAE metrics.

## Tech Stack
- **Frontend:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest, Gradient Boosting)
- **Data Visualization:** Seaborn, Matplotlib

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/harshitha588/Insurance-Charges-Predictor.git
   cd Insurance-Charges-Predictor
   ```

2. **Install Required Packages**
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install streamlit pandas numpy seaborn matplotlib scikit-learn
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

4. **Access the Dashboard**
   Open your browser and navigate to `http://localhost:8501`.

## Dataset
The project uses the standard `insurance.csv` dataset, which includes attributes like age, sex, bmi, children, smoker, region, and charges.

## Design
Features a premium UI built with custom CSS for engaging data presentation, glassmorphism cards, and responsive layouts.
>>>>>>> 83c3513 (Initial commit)
