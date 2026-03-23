import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI-Based Multi-Disease Prediction System")
st.markdown("### Predict Diabetes | Heart Disease | Parkinson's Disease")
st.success("✅ System Ready! Enter your health details below.")

# Define prediction functions
def predict_diabetes(pregnancies, glucose, bmi, age):
    """Simple rule-based diabetes risk assessment"""
    risk_score = 0
    if glucose > 120:
        risk_score += 40
    if bmi > 25:
        risk_score += 30
    if age > 40:
        risk_score += 20
    if pregnancies > 2:
        risk_score += 10
    
    if risk_score >= 60:
        return "HIGH RISK", risk_score
    else:
        return "LOW RISK", risk_score

def predict_heart(age, bp, cholesterol, max_hr):
    """Simple rule-based heart disease risk assessment"""
    risk_score = 0
    if age > 50:
        risk_score += 30
    if bp > 130:
        risk_score += 30
    if cholesterol > 200:
        risk_score += 30
    if max_hr < 150:
        risk_score += 10
    
    if risk_score >= 60:
        return "HIGH RISK", risk_score
    else:
        return "LOW RISK", risk_score

def predict_parkinsons(jitter, shimmer, hnr):
    """Simple rule-based Parkinson's risk assessment"""
    risk_score = 0
    if jitter > 0.01:
        risk_score += 40
    if shimmer > 0.05:
        risk_score += 40
    if hnr < 15:
        risk_score += 20
    
    if risk_score >= 60:
        return "HIGH LIKELIHOOD", risk_score
    else:
        return "LOW LIKELIHOOD", risk_score

# Tabs
tab1, tab2, tab3 = st.tabs(["🩺 Diabetes", "❤️ Heart Disease", "🧠 Parkinson's"])

# ============================================
# DIABETES TAB
# ============================================
with tab1:
    st.subheader("Diabetes Risk Assessment")
    st.markdown("Enter your medical details:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=120)
    
    with col2:
        bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=45.0, value=25.0, format="%.1f")
        age = st.number_input("Age (years)", min_value=20, max_value=100, value=30)
    
    if st.button("🔮 Predict Diabetes", type="primary"):
        result, risk = predict_diabetes(pregnancies, glucose, bmi, age)
        
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        # Show risk meter
        st.markdown(f"**Risk Score:** {risk}/100")
        st.progress(risk/100)
        
        if "HIGH" in result:
            st.error(f"⚠️ **{result} of Diabetes**")
            st.warning("💡 Please consult a doctor for proper diagnosis and management.")
        else:
            st.success(f"✅ **{result} of Diabetes**")
            st.info("💡 Keep maintaining a healthy lifestyle!")

# ============================================
# HEART TAB
# ============================================
with tab2:
    st.subheader("Heart Disease Risk Assessment")
    st.markdown("Enter your clinical measurements:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_h = st.number_input("Age (years)", min_value=20, max_value=100, value=50, key="heart_age")
        bp_h = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
    
    with col2:
        cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200)
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    
    if st.button("🔮 Predict Heart Disease", type="primary"):
        result, risk = predict_heart(age_h, bp_h, cholesterol, max_hr)
        
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        # Show risk meter
        st.markdown(f"**Risk Score:** {risk}/100")
        st.progress(risk/100)
        
        if "HIGH" in result:
            st.error(f"⚠️ **{result} of Heart Disease**")
            st.warning("💡 Please consult a cardiologist for further evaluation.")
        else:
            st.success(f"✅ **{result} of Heart Disease**")
            st.info("💡 Maintain a healthy lifestyle with regular exercise!")

# ============================================
# PARKINSON'S TAB
# ============================================
with tab3:
    st.subheader("Parkinson's Disease Risk Assessment")
    st.info("🎤 Voice measurements from standardized test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        jitter = st.slider("Jitter (%)", min_value=0.0, max_value=0.1, value=0.006, format="%.4f", 
                          help="Higher values indicate voice instability")
        shimmer = st.slider("Shimmer", min_value=0.0, max_value=0.2, value=0.03, format="%.3f",
                           help="Higher values indicate voice amplitude variation")
    
    with col2:
        hnr = st.slider("Harmonic-to-Noise Ratio (HNR)", min_value=0.0, max_value=30.0, value=20.0,
                        help="Lower values indicate more noise in voice")
    
    if st.button("🔮 Predict Parkinson's Disease", type="primary"):
        result, risk = predict_parkinsons(jitter, shimmer, hnr)
        
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        # Show risk meter
        st.markdown(f"**Risk Score:** {risk}/100")
        st.progress(risk/100)
        
        if "HIGH" in result:
            st.error(f"⚠️ **{result} of Parkinson's Disease**")
            st.warning("💡 Please consult a neurologist for proper evaluation.")
        else:
            st.success(f"✅ **{result} of Parkinson's Disease**")
            st.info("💡 Continue maintaining good vocal health!")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About This System")
    st.markdown("""
    ### 🏥 **Multi-Disease Prediction System**
    
    Predicts:
    - **Diabetes** - Based on glucose, BMI, age, pregnancies
    - **Heart Disease** - Based on age, BP, cholesterol, heart rate  
    - **Parkinson's Disease** - Based on voice measurements
    
    ### 📊 **Risk Calculation**
    Uses simple rule-based scoring:
    - **0-30%**: Low Risk
    - **31-60%**: Moderate Risk
    - **61-100%**: High Risk
    
    ### 🔬 **Risk Factors**
    
    **Diabetes:**
    - Glucose > 120 mg/dL: +40%
    - BMI > 25: +30%
    - Age > 40: +20%
    - Pregnancies > 2: +10%
    
    **Heart Disease:**
    - Age > 50: +30%
    - BP > 130: +30%
    - Cholesterol > 200: +30%
    - Max HR < 150: +10%
    
    **Parkinson's:**
    - Jitter > 1%: +40%
    - Shimmer > 5%: +40%
    - HNR < 15: +20%
    
    ### ⚠️ **Disclaimer**
    This is for **educational purposes only**.
    Always consult healthcare professionals for medical advice.
    """)
    
    st.markdown("---")
    st.caption("Final Year Project | AI/ML | Multi-Disease Prediction")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>AI-Based Multi-Disease Prediction System | For Educational Purposes Only</p>", unsafe_allow_html=True)
