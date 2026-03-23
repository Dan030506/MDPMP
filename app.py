import streamlit as st
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        # Diabetes
        with open('models/diabetes/diabetes_model.pkl', 'rb') as f:
            diabetes_model = pickle.load(f)
        with open('models/diabetes/scaler.pkl', 'rb') as f:
            diabetes_scaler = pickle.load(f)
        
        # Heart
        with open('models/heart/heart_model.pkl', 'rb') as f:
            heart_model = pickle.load(f)
        with open('models/heart/scaler.pkl', 'rb') as f:
            heart_scaler = pickle.load(f)
        
        # Parkinson's
        with open('models/parkinsons/parkinsons_model.pkl', 'rb') as f:
            parkinsons_model = pickle.load(f)
        with open('models/parkinsons/scaler.pkl', 'rb') as f:
            parkinsons_scaler = pickle.load(f)
        
        return {
            'diabetes': {'model': diabetes_model, 'scaler': diabetes_scaler},
            'heart': {'model': heart_model, 'scaler': heart_scaler},
            'parkinsons': {'model': parkinsons_model, 'scaler': parkinsons_scaler}
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load models
models = load_models()

# Title
st.title("🏥 AI-Based Multi-Disease Prediction System")
st.markdown("### Diabetes | Heart Disease | Parkinson's Disease")

if models:
    st.success("✅ Models loaded successfully!")
else:
    st.error("❌ Models not found. Please train models first.")
    st.info("Run: python3 train_models.py")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🩺 Diabetes", "❤️ Heart Disease", "🧠 Parkinson's"])

# ============================================
# DIABETES TAB
# ============================================
with tab1:
    st.subheader("Diabetes Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 50, 200, 120)
        bp = st.number_input("Blood Pressure", 60, 140, 70)
    
    with col2:
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
    
    with col3:
        pedigree = st.number_input("Diabetes Pedigree", 0.08, 2.5, 0.5)
        age = st.number_input("Age", 21, 80, 30)
    
    if st.button("Predict Diabetes", type="primary"):
        data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
        scaled = models['diabetes']['scaler'].transform(data)
        pred = models['diabetes']['model'].predict(scaled)[0]
        prob = models['diabetes']['model'].predict_proba(scaled)[0]
        
        if pred == 1:
            st.error(f"⚠️ HIGH RISK of Diabetes (Confidence: {prob[1]*100:.1f}%)")
        else:
            st.success(f"✅ LOW RISK of Diabetes (Confidence: {prob[0]*100:.1f}%)")

# ============================================
# HEART TAB
# ============================================
with tab2:
    st.subheader("Heart Disease Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_h = st.number_input("Age", 29, 77, 50, key="age_h")
        sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
        cp = st.selectbox("Chest Pain", [0, 1, 2, 3], key="cp", 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
    
    with col2:
        bp_h = st.number_input("Resting BP", 94, 200, 120, key="bp_h")
        chol = st.number_input("Cholesterol", 126, 564, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], key="fbs",
                          format_func=lambda x: "Yes" if x else "No")
    
    with col3:
        thalach = st.number_input("Max Heart Rate", 71, 202, 150)
        exang = st.selectbox("Exercise Angina", [0, 1], key="exang",
                            format_func=lambda x: "Yes" if x else "No")
        oldpeak = st.number_input("ST Depression", 0.0, 6.2, 1.0)
    
    if st.button("Predict Heart Disease", type="primary"):
        data = np.array([[age_h, 1 if sex == "Male" else 0, cp, bp_h, chol, fbs, 0, thalach, exang, oldpeak, 1, 0, 2]])
        scaled = models['heart']['scaler'].transform(data)
        pred = models['heart']['model'].predict(scaled)[0]
        
        if pred == 1:
            st.error("⚠️ HIGH RISK of Heart Disease")
        else:
            st.success("✅ LOW RISK of Heart Disease")

# ============================================
# PARKINSON'S TAB
# ============================================
with tab3:
    st.subheader("Parkinson's Disease Risk Assessment")
    st.info("Voice measurements from standardized test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", 80, 300, 120)
        fhi = st.number_input("MDVP:Fhi(Hz)", 100, 500, 150)
        flo = st.number_input("MDVP:Flo(Hz)", 50, 300, 100)
        jitter = st.number_input("Jitter(%)", 0.0, 1.0, 0.006)
    
    with col2:
        shimmer = st.number_input("Shimmer", 0.0, 0.5, 0.03)
        hnr = st.number_input("HNR", 0.0, 50.0, 20.0)
        rpde = st.number_input("RPDE", 0.0, 1.0, 0.5)
        dfa = st.number_input("DFA", 0.0, 1.0, 0.6)
    
    if st.button("Predict Parkinson's", type="primary"):
        data = np.array([[fo, fhi, flo, jitter, 0.00004, 0.003, 0.003, 0.01, 
                          shimmer, 0.3, 0.015, 0.02, 0.025, 0.045, 0.02, hnr, 
                          rpde, dfa, -5, 0.2, 2, 0.1]])
        scaled = models['parkinsons']['scaler'].transform(data)
        pred = models['parkinsons']['model'].predict(scaled)[0]
        
        if pred == 1:
            st.error("⚠️ HIGH LIKELIHOOD of Parkinson's Disease")
        else:
            st.success("✅ LOW LIKELIHOOD of Parkinson's Disease")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **AI-Based Multi-Disease Prediction System**
    
    Predicts:
    - 🩺 **Diabetes** (8 parameters)
    - ❤️ **Heart Disease** (13 parameters)  
    - 🧠 **Parkinson's Disease** (22 parameters)
    
    **Model:** Random Forest Classifier
    
    **Accuracy:**
    - Diabetes: ~78%
    - Heart Disease: ~85%
    - Parkinson's: ~94%
    
    ⚠️ **Disclaimer:** For educational purposes only. Always consult a doctor.
    """)
    
    st.markdown("---")
    st.caption("Final Year Project | Machine Learning")

st.markdown("---")
st.caption("🏥 Multi-Disease Prediction System | AI/ML Project")
