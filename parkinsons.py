import streamlit as st
import shap
from streamlit_shap import st_shap
import numpy as np
import pandas as pd

def inject_custom_css():
    # Neurology/Brain abstract background
    # background_image_url = "https://img.freepik.com/premium-photo/blue-motor-neuron-nerve-cell-with-dark-background-3d-rendering_648796-1300.jpg"
    
    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    [data-testid="stAppViewContainer"] {{
        background: radial-gradient(circle at 50% 50%, #1a0b2e 0%, #000000 100%);
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"] {{
        
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"] > .main {{
        background-color: rgba(0, 0, 0, 0.85); backdrop-filter: blur(10px);
    }}
    h1, h2, h3, p, span, label {{ color: #F0F2F6 !important; }}
    .stButton > button {{
        background: linear-gradient(135deg, #8A2387 0%, #E94057 50%, #F27121 100%);
        color: white; border: none; border-radius: 8px; padding: 10px 24px; font-weight: 600;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0, 198, 255, 0.4); width: 100%;
    }}
    .stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 114, 255, 0.6); color: white; }}
    .stTextInput > div > div > input, .stNumberInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.1); color: white; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 5px;
    }}
    iframe {{ background-color: rgba(255, 255, 255, 0.95); border-radius: 10px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
    h3 {{ margin-top: 20px; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 5px; }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def run(models, scalers, backgrounds):
    inject_custom_css()
    st.title("🧠 Parkinson's Disease Prediction")
    st.write("Enter the patient's acoustic vocal biomarker measurements below:")

    # --- CLINICAL LAYOUT GROUPING (Now starting at 0.0) ---
    st.subheader("1. Vocal Fundamental Frequency Parameters")
    col1, col2, col3 = st.columns(3)
    with col1: fo = st.number_input('MDVP:Fo(Hz) (Average)', value=0.0, format="%.5f")
    with col2: fhi = st.number_input('MDVP:Fhi(Hz) (Maximum)', value=0.0, format="%.5f")
    with col3: flo = st.number_input('MDVP:Flo(Hz) (Minimum)', value=0.0, format="%.5f")

    st.subheader("2. Jitter (Frequency Variation)")
    col4, col5, col6, col7, col8 = st.columns(5)
    with col4: jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.0, format="%.5f")
    with col5: jitter_abs = st.number_input('MDVP:Jitter(Abs)', value=0.0, format="%.6f")
    with col6: rap = st.number_input('MDVP:RAP', value=0.0, format="%.5f")
    with col7: ppq = st.number_input('MDVP:PPQ', value=0.0, format="%.5f")
    with col8: ddp = st.number_input('Jitter:DDP', value=0.0, format="%.5f")

    st.subheader("3. Shimmer (Amplitude Variation)")
    col9, col10, col11, col12, col13, col14 = st.columns(6)
    with col9: shimmer = st.number_input('MDVP:Shimmer', value=0.0, format="%.5f")
    with col10: shimmer_db = st.number_input('MDVP:Shimmer(dB)', value=0.0, format="%.5f")
    with col11: apq3 = st.number_input('Shimmer:APQ3', value=0.0, format="%.5f")
    with col12: apq5 = st.number_input('Shimmer:APQ5', value=0.0, format="%.5f")
    with col13: apq = st.number_input('MDVP:APQ', value=0.0, format="%.5f")
    with col14: dda = st.number_input('Shimmer:DDA', value=0.0, format="%.5f")

    st.subheader("4. Noise & Nonlinear Dynamics")
    col15, col16, col17, col18 = st.columns(4)
    with col15: nhr = st.number_input('NHR (Noise-to-Harmonic)', value=0.0, format="%.5f")
    with col16: hnr = st.number_input('HNR (Harmonic-to-Noise)', value=0.0, format="%.5f")
    with col17: rpde = st.number_input('RPDE (Complexity)', value=0.0, format="%.5f")
    with col18: dfa = st.number_input('DFA (Fractal Scaling)', value=0.0, format="%.5f")

    col19, col20, col21, col22 = st.columns(4)
    with col19: spread1 = st.number_input('spread1', value=0.0, format="%.5f")
    with col20: spread2 = st.number_input('spread2', value=0.0, format="%.5f")
    with col21: d2 = st.number_input('D2 (Correlation Dim)', value=0.0, format="%.5f")
    with col22: ppe = st.number_input('PPE (Pitch Period)', value=0.0, format="%.5f")

    st.markdown("<br>", unsafe_allow_html=True) 

    if st.button("Run Neurological Diagnostic"):

        # 1. MATCH THE EXACT ORDER OF THE MODEL
        feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
                         'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
                         'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
                         'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

        user_input = [[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, 
                       apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]]

        # 2. Convert to DataFrame
        input_df = pd.DataFrame(user_input, columns=feature_names)

        # 3. Safely extract model and scaler
        actual_scaler = scalers.get('parkinsons') or list(scalers.values())[2]
        actual_model = models.get('parkinsons') or list(models.values())[2]

        # 4. Scale Input (Keep as a raw NumPy array to match the SVM's training data)
        input_scaled = actual_scaler.transform(input_df)
        
        # 5. Make Prediction
        parkinsons_prediction = actual_model.predict(input_scaled)
        
        st.markdown("---")
        if parkinsons_prediction[0] == 1:
            st.error("⚠️ High Risk: Acoustic biomarkers indicate the presence of Parkinson's disease.") 
        else:
            st.success("✅ Low Risk: Acoustic biomarkers do not indicate Parkinson's disease.") 

        with st.expander("Diagnostic Feature Analysis (SHAP)", expanded=True):
            st.write("Red features push the model toward a positive diagnosis, while blue features push it toward a negative (healthy) diagnosis.")
            with st.spinner("Generating AI explanation (this takes a few seconds)..."):
                
                # Safely extract background for KernelExplainer
                # Safely extract background for KernelExplainer without triggering truth-value errors
                actual_bg = backgrounds.get('parkinsons')
                if actual_bg is None:
                    actual_bg = list(backgrounds.values())[0]
                    
                explainer = shap.KernelExplainer(actual_model.predict, actual_bg)
                raw_shap_values = explainer.shap_values(input_scaled)
                
                # The Bulletproof Flatten Method
                base_val = np.array(explainer.expected_value).flatten()[-1]
                shap_vals = np.array(raw_shap_values).flatten()[-len(feature_names):]
                
                # Pass the RAW input_df so SHAP displays real human numbers!
                force_plot = shap.force_plot(
                    base_val, 
                    shap_vals, 
                    input_df.iloc[0], 
                    feature_names=feature_names
                )
                
                st_shap(force_plot, height=150)