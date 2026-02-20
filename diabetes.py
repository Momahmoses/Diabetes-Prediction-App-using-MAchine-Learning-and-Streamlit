import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------------------------------------------
# PART 1: MODEL TRAINING & CONFIGURATION
# -----------------------------------------------------------------------------

# We use @st.cache_resource so Streamlit runs this function ONLY ONCE.
# This prevents the app from retraining the model on every user interaction.
@st.cache_resource
def build_model():
    """
    Loads data, processes it, and trains the Random Forest model.
    Returns: The trained model, the list of expected columns, and the accuracy score.
    """
    try:
        # 1. Load Data
        df = pd.read_csv('diabetes_dataset.csv')
        
        # 2. Data Cleaning
        # Remove unneeded index column if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            
        # Fill missing values for Alcohol with 'None'
        df['Alcohol_Consumption'] = df['Alcohol_Consumption'].fillna('None')

        # 3. Create Target Variable (Risk Definition)
        # Diabetic if Fasting Glucose >= 126 OR HbA1c >= 6.5
        df['Diabetes_Risk'] = ((df['Fasting_Blood_Glucose'] >= 126) | 
                               (df['HbA1c'] >= 6.5)).astype(int)

        # 4. Define Features (X) and Target (y)
        # CRITICAL: We drop the variables used to define the target (Glucose/HbA1c)
        # to prevent "data leakage". We want to predict risk based on external signs.
        X = df.drop(columns=['Diabetes_Risk', 'Fasting_Blood_Glucose', 'HbA1c'])
        y = df['Diabetes_Risk']

        # 5. Encoding Categorical Variables
        # Convert text columns (Sex, Ethnicity, etc.) into numbers (0s and 1s)
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Save the column names to ensure user input matches exactly later
        model_columns = X_encoded.columns

        # 6. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # 7. Model Training
        # We use class_weight='balanced' to handle the fact that there are fewer
        # non-diabetic cases in this specific dataset.
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # 8. Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, model_columns, accuracy

    except FileNotFoundError:
        return None, None, None

# Run the training function
model, model_columns, accuracy = build_model()

# -----------------------------------------------------------------------------
# PART 2: STREAMLIT USER INTERFACE
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# Header Section
st.title("🩺 Intelligent Diabetes Risk Prediction System")
st.markdown("""
This application uses a **Random Forest Machine Learning model** to assess the probability of diabetes 
based on health indicators.  
*Data Source: Internal Clinical Dataset | Model Accuracy: {:.1f}%*
""".format(accuracy * 100 if accuracy else 0))

# Check if model loaded correctly
if model is None:
    st.error("⚠️ Error: 'diabetes_dataset.csv' not found. Please place the dataset in the same directory.")
    st.stop()

st.divider()

# Input Form
st.subheader("Patient Information Input")

# Create a 3-column layout for better UI
col1, col2, col3 = st.columns(3)

with col1:
    st.info("Personal Details")
    age = st.number_input("Age (Years)", 18, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"])
    family_hist = st.selectbox("Family History of Diabetes", ["No", "Yes"])
    gest_diab = st.selectbox("History of Gestational Diabetes", ["No", "Yes"])

with col2:
    st.info("Physical Health")
    bmi = st.number_input("BMI", 15.0, 60.0, 28.5)
    waist = st.number_input("Waist Circumference (cm)", 50.0, 150.0, 95.0)
    bp_sys = st.number_input("Systolic BP (mm Hg)", 90, 200, 120)
    bp_dia = st.number_input("Diastolic BP (mm Hg)", 50, 130, 80)

with col3:
    st.info("Lifestyle & Labs")
    chol_total = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
    phys_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
    diet_cal = st.number_input("Daily Calories", 1000, 5000, 2500)
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])

# Predict Button
if st.button("🔍 Analyze Risk Profile", type="primary"):
    
    # -------------------------------------------------------------------------
    # PART 3: PREDICTION LOGIC
    # -------------------------------------------------------------------------
    
    # 1. Gather Input into a Dictionary
    input_data = {
        'Age': age,
        'Sex': sex,
        'Ethnicity': ethnicity,
        'BMI': bmi,
        'Waist_Circumference': waist,
        'Blood_Pressure_Systolic': bp_sys,
        'Blood_Pressure_Diastolic': bp_dia,
        'Cholesterol_Total': chol_total,
        # Defaulting missing lab values to averages as they weren't in the inputs
        'Cholesterol_HDL': 50, 
        'Cholesterol_LDL': 100,
        'GGT': 30,
        'Serum_Urate': 5,
        'Physical_Activity_Level': phys_activity,
        'Dietary_Intake_Calories': diet_cal,
        'Alcohol_Consumption': alcohol,
        'Smoking_Status': smoking,
        'Family_History_of_Diabetes': 1 if family_hist == "Yes" else 0,
        'Previous_Gestational_Diabetes': 1 if gest_diab == "Yes" else 0
    }

    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. One-Hot Encoding (converting user text choices to 0/1 columns)
    input_encoded = pd.get_dummies(input_df)

    # 4. Alignment
    # The user input might be missing columns (e.g., if they chose 'Male', 
    # there is no 'Sex_Female' column created). We enforce the model's structure here.
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    # 5. Prediction
    # probability[1] is the chance of being positive (Diabetic)
    probability = model.predict_proba(input_final)[0][1]
    
    # 6. Result Display
    st.divider()
    st.subheader("Analysis Report")
    
    # Create a visual progress bar for risk
    st.progress(probability)
    
    col_res1, col_res2 = st.columns([2, 1])
    
    with col_res1:
        if probability >= 0.7:
            st.error(f"**High Risk Detected** ({probability:.1%} probability)")
            st.write("The model suggests a high likelihood of diabetic indicators based on the provided metrics.")
        elif probability >= 0.4:
            st.warning(f"**Moderate Risk Detected** ({probability:.1%} probability)")
            st.write("There are elevated risk factors. Lifestyle adjustments are recommended.")
        else:
            st.success(f"**Low Risk Detected** ({probability:.1%} probability)")
            st.write("The input profile does not show strong indicators of diabetes risk.")
            
    with col_res2:
        st.caption("Contributing Factors (Feature Importance)")
        # Show top 3 factors driving the decision
        importances = pd.Series(model.feature_importances_, index=model_columns).sort_values(ascending=False)
        st.bar_chart(importances.head(3))