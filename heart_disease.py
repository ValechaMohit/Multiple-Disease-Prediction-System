import streamlit as st
import shap
from streamlit_shap import st_shap
import numpy as np
import pandas as pd

def inject_custom_css():
    # Abstract dark red/cardiology background
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 50%, #2b0305 0%, #000000 100%);
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"] > .main {
        background-color: rgba(0, 0, 0, 0.85); backdrop-filter: blur(10px);
    }
    h1, h2, h3, p, span, label { color: #F0F2F6 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #8A2387 0%, #E94057 50%, #F27121 100%);
        color: white; border: none; border-radius: 8px; padding: 10px 24px; font-weight: 600;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4); width: 100%;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(255, 75, 43, 0.6); color: white; }
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.1); color: white; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 5px;
    }
    iframe { background-color: rgba(255, 255, 255, 0.95); border-radius: 10px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def run(models, scalers):
    inject_custom_css()
    st.title('🫀 Heart Disease Prediction')
    st.write("Enter the patient's vitals and test results below:")

    # Define layout
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=1)
        sex_ui = st.selectbox('Sex', ['Female', 'Male'])
        cp_ui = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=50)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=100)

    with col2:
        fbs_ui = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
        restecg_ui = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=250, value=60)
        exang_ui = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=10.0, value=0.0, step=0.1)

    with col3:
        slope_ui = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
        thal_ui = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversable Defect'])

    if st.button('Run Cardiology Diagnostic'):
        
        # --- MAP UI TEXT TO MODEL NUMBERS ---
        sex = 1 if sex_ui == 'Male' else 0
        cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp_ui)
        fbs = 1 if fbs_ui == 'True' else 0
        restecg = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg_ui)
        exang = 1 if exang_ui == 'Yes' else 0
        slope = ['Upsloping', 'Flat', 'Downsloping'].index(slope_ui)
        thal_dict = {'Normal': 1, 'Fixed Defect': 2, 'Reversable Defect': 3}
        thal = thal_dict[thal_ui]

        # 1. MATCH THE EXACT ORDER OF THE MODEL
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        user_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        
        # 2. Convert to DataFrame
        input_df = pd.DataFrame(user_input, columns=feature_names)
        
        # 3. Safely extract model and scaler from app.py
        actual_scaler = scalers.get('heart_disease') or list(scalers.values())[1]  # Safety fallback
        actual_model = models.get('heart_disease') or list(models.values())[1]

        # 4. --- SELECTIVE SCALING ---
        # We only scale the specific columns the scaler was trained on!
        cols_to_scale = ['trestbps', 'chol', 'thalach', 'oldpeak']
        input_scaled = input_df.copy()
        input_scaled[cols_to_scale] = actual_scaler.transform(input_df[cols_to_scale])
        
        # 5. Make Prediction
        heart_prediction = actual_model.predict(input_scaled)
        
        if heart_prediction[0] == 1:
            st.error('⚠️ The Patient is at High Risk for Heart Disease.')
        else:
            st.success('✅ The Patient is at Low Risk for Heart Disease.')
            
        with st.expander("Diagnostic Feature Analysis (SHAP)", expanded=True):
            st.write("Features pushing the prediction closer to disease are in red, while those pushing it closer to healthy are in blue.")
            with st.spinner("Generating AI explanation..."):
                explainer = shap.TreeExplainer(actual_model)
                shap_values = explainer.shap_values(input_scaled)
                
                # Format base value and SHAP values correctly
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                shap_val = shap_values[1][0] if isinstance(shap_values, list) else (shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0])
                
                # IMPORTANT: Pass the RAW input_df so SHAP displays real human numbers, not the scaled decimals!
                st_shap(shap.force_plot(base_val, shap_val, input_df.iloc[0], feature_names=feature_names))