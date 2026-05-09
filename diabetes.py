import streamlit as st
import shap
from streamlit_shap import st_shap
import numpy as np
import pandas as pd

def inject_custom_css():
    # Abstract dark purple/blue background for Diabetes
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 50%, #1a0b2e 0%, #000000 100%);
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"] > .main {
        background-color: rgba(0, 0, 0, 0.85); backdrop-filter: blur(10px);
    }
    h1, h2, h3, p, span, label { color: #F0F2F6 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #8A2387 0%, #E94057 50%, #F27121 100%);
        color: white; border: none; border-radius: 8px; padding: 10px 24px; font-weight: 600;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(233, 64, 87, 0.4); width: 100%;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(233, 64, 87, 0.6); color: white; }
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.1); color: white; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 5px;
    }
    iframe { background-color: rgba(255, 255, 255, 0.95); border-radius: 10px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def run(models, scalers):
    inject_custom_css()
    st.title('🩸 Diabetes Prediction')
    st.write("Enter the patient's vitals and test results below:")

    # Define a clean, balanced layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=45)
        glucose = st.number_input('Fasting Plasma Glucose (mg/dL)', min_value=0, max_value=300, value=110)
        bmi = st.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
    with col2:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
        blood_pressure = st.number_input('Systolic Blood Pressure (mmHg)', min_value=50, max_value=200, value=120)
        family_risk_ui = st.selectbox('Family History Risk', ['Low Risk', 'Medium Risk', 'High Risk'])

    if st.button('Run Diabetes Diagnostic'):
        
        # --- MAP UI TEXT TO MODEL NUMBERS ---
        risk_map = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}
        family_risk = risk_map[family_risk_ui]
        
        # 1. MATCH THE EXACT ORDER OF THE MODEL
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age', 'Family_History_Risk']
        user_input = [[pregnancies, glucose, blood_pressure, bmi, age, family_risk]]
        
        # 2. Convert to DataFrame
        input_df = pd.DataFrame(user_input, columns=feature_names)
        
        # 3. Safely extract model and scaler
        actual_scaler = scalers.get('diabetes') or scalers.get('diabetes_scaler') or list(scalers.values())[0]
        actual_model = models.get('diabetes') or models.get('diabetes_model') or list(models.values())[0]

        # 4. --- SELECTIVE SCALING (THE FIX IS HERE) ---
        cols_to_scale = ['Glucose', 'BloodPressure', 'BMI']
        input_scaled = input_df.copy()
        input_scaled[cols_to_scale] = actual_scaler.transform(input_df[cols_to_scale])
        
        # 5. Make Prediction
        diab_prediction = actual_model.predict(input_scaled)
        
        if diab_prediction[0] == 1:
            st.error('⚠️ The patient is at high risk of having diabetes.')
        else:
            st.success('✅ The patient is not at high risk of having diabetes.')
            
        with st.expander("Diagnostic Feature Analysis (SHAP)", expanded=True):
            st.write("Features pushing the prediction closer to disease are in red, while those pushing it closer to healthy are in blue.")
            with st.spinner("Generating AI explanation..."):
                explainer = shap.TreeExplainer(actual_model)
                shap_values = explainer.shap_values(input_scaled)
                
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                shap_val = shap_values[1][0] if isinstance(shap_values, list) else (shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0])
                
                # IMPORTANT: Pass the RAW input_df so SHAP displays real human numbers!
                st_shap(shap.force_plot(base_val, shap_val, input_df.iloc[0], feature_names=feature_names))