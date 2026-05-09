import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import diabetes
import heart_disease
import parkinsons


st.set_page_config(
    page_title="Multiple Disease Prediction", 
    page_icon="⚕️", 
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_resource
def load_assets():
    models = {
        'diabetes_model' : pickle.load(open('diabetes_model.sav', 'rb')),
        'heart_disease': pickle.load(open('heart_disease.sav', 'rb')), # or heart_disease_model.sav depending on your file name
        'parkinsons': pickle.load(open('parkinsons_model.sav', 'rb')),
    }
    scalers = {
        'diabetes_scaler': pickle.load(open('diabetes_scaler.sav', 'rb')),
        'heart_disease': pickle.load(open('heart_scaler.sav', 'rb')), # or heart_disease_scaler.sav depending on your file name
        'parkinsons': pickle.load(open('parkinsons_scaler.sav', 'rb')), # 🚨 RESTORED!
    }
    backgrounds = {
        'parkinsons': pickle.load(open('parkinsons_bg.sav', 'rb')) # 🚨 RESTORED!
    }
    return models, scalers, backgrounds
    
   

try:
    models, scalers, backgrounds = load_assets()
except FileNotFoundError as e:
    st.error(f"🚨 Missing File Error: {e}")
    st.info("Please ensure your .sav files are in the exact same folder as app.py")
    st.stop()


with st.sidebar:
    selected = option_menu(
        'Diagnostic Modules',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['droplet', 'heart-pulse', 'person-lines-fill'],
        menu_icon='hospital-fill',
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#0e1117"},
            "icon": {"color": "#00C6FF", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#262730", "color": "#fafafa"},
            "nav-link-selected": {"background-color": "#0072FF"},
        }
    )


if selected == 'Diabetes Prediction':
    diabetes.run(models, scalers)
elif selected == 'Heart Disease Prediction':
    heart_disease.run(models, scalers)
elif selected == 'Parkinsons Prediction':
    
    parkinsons.run(models, scalers, backgrounds)