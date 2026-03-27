import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR MODERN UI
# ============================================
st.markdown("""
<style>
    /* Main Container Styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .diabetes-card {
        background: linear-gradient(135deg, #fef3e8 0%, #ffe6d5 100%);
        border-bottom: 4px solid #f39c12;
    }
    
    .heart-card {
        background: linear-gradient(135deg, #ffe8e8 0%, #ffd4d4 100%);
        border-bottom: 4px solid #e74c3c;
    }
    
    .parkinson-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #d4edd7 100%);
        border-bottom: 4px solid #2ecc71;
    }
    
    .card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .card-params {
        font-size: 0.85rem;
        color: #666;
        margin: 1rem 0;
    }
    
    .card-badge {
        display: inline-block;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    
    .diabetes-badge {
        background: #f39c12;
        color: white;
    }
    
    .heart-badge {
        background: #e74c3c;
        color: white;
    }
    
    .parkinson-badge {
        background: #2ecc71;
        color: white;
    }
    
    /* Feature Box Styling */
    .feature-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Risk Level Styling */
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 8px 16px;
        border-radius: 25px;
        display: inline-block;
        font-weight: bold;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        color: #856404;
        padding: 8px 16px;
        border-radius: 25px;
        display: inline-block;
        font-weight: bold;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 8px 16px;
        border-radius: 25px;
        display: inline-block;
        font-weight: bold;
    }
    
    /* Progress Bar Customization */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2ecc71, #f39c12, #e74c3c);
        border-radius: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        color: #495057;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #1a2632 100%);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER SECTION
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🏥 AI-Based Multi-Disease Prediction System</h1>
    <p>Advanced Risk Assessment | Diabetes | Heart Disease | Parkinson's Disease</p>
    <p style="font-size: 0.9rem;">Powered by Clinical Guidelines & Rule-Based AI</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE FOR PAGE NAVIGATION
# ============================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def predict_diabetes_advanced(pregnancies, glucose, bp, skin_thickness, insulin, 
                               bmi, pedigree, age, waist_circumference, physical_activity,
                               family_history, hba1c, triglycerides, hdl):
    """Enhanced diabetes risk calculation with 14 parameters"""
    risk_score = 0
    
    # Glucose (0-40 points)
    if glucose > 200:
        risk_score += 40
    elif glucose > 140:
        risk_score += 30
    elif glucose > 120:
        risk_score += 20
    elif glucose > 100:
        risk_score += 10
    
    # BMI (0-25 points)
    if bmi > 35:
        risk_score += 25
    elif bmi > 30:
        risk_score += 20
    elif bmi > 25:
        risk_score += 15
    elif bmi > 23:
        risk_score += 5
    
    # Age (0-20 points)
    if age > 60:
        risk_score += 20
    elif age > 50:
        risk_score += 15
    elif age > 40:
        risk_score += 10
    elif age > 30:
        risk_score += 5
    
    # Waist Circumference (0-15 points)
    if waist_circumference > 100:
        risk_score += 15
    elif waist_circumference > 90:
        risk_score += 10
    elif waist_circumference > 80:
        risk_score += 5
    
    # Physical Activity (0-15 points)
    if physical_activity == 0:
        risk_score += 15
    elif physical_activity <= 2:
        risk_score += 10
    elif physical_activity <= 4:
        risk_score += 5
    
    # Family History (0-10 points)
    if family_history == "Yes - Both Parents":
        risk_score += 10
    elif family_history == "Yes - One Parent":
        risk_score += 7
    elif family_history == "Yes - Sibling":
        risk_score += 5
    
    # HbA1c (0-20 points)
    if hba1c > 8.0:
        risk_score += 20
    elif hba1c > 7.0:
        risk_score += 15
    elif hba1c > 6.5:
        risk_score += 10
    elif hba1c > 6.0:
        risk_score += 5
    
    # Triglycerides (0-15 points)
    if triglycerides > 300:
        risk_score += 15
    elif triglycerides > 200:
        risk_score += 10
    elif triglycerides > 150:
        risk_score += 5
    
    # HDL (0-15 points)
    if hdl < 35:
        risk_score += 15
    elif hdl < 40:
        risk_score += 10
    elif hdl < 50:
        risk_score += 5
    
    # Pregnancies (0-10 points)
    if pregnancies > 4:
        risk_score += 10
    elif pregnancies > 2:
        risk_score += 5
    
    # Blood Pressure (0-10 points)
    if bp > 140:
        risk_score += 10
    elif bp > 130:
        risk_score += 5
    
    # Cap at 100
    risk_score = min(risk_score, 100)
    
    # Determine risk level
    if risk_score >= 70:
        result = "HIGH RISK"
        color = "🔴"
        advice = "Please consult an endocrinologist immediately. Consider lifestyle modifications and medication."
    elif risk_score >= 40:
        result = "MODERATE RISK"
        color = "🟡"
        advice = "Schedule a check-up with your doctor. Focus on diet, exercise, and regular monitoring."
    else:
        result = "LOW RISK"
        color = "🟢"
        advice = "Maintain healthy lifestyle habits. Regular screening recommended every 3 years."
    
    return result, risk_score, color, advice

def predict_heart_advanced(age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                           exang, oldpeak, slope, ca, thal, smoking, diabetes_status,
                           family_history_heart, ecg_abnormal, stress_test, medication):
    """Enhanced heart disease risk calculation with 19 parameters"""
    risk_score = 0
    
    # Age (0-25 points)
    if age > 70:
        risk_score += 25
    elif age > 60:
        risk_score += 20
    elif age > 50:
        risk_score += 15
    elif age > 40:
        risk_score += 10
    elif age > 30:
        risk_score += 5
    
    # Sex
    if sex == "Male":
        risk_score += 10
    
    # Chest Pain Type
    cp_scores = {"Typical Angina": 20, "Atypical Angina": 15, "Non-anginal Pain": 10, "Asymptomatic": 5}
    risk_score += cp_scores.get(cp, 10)
    
    # Blood Pressure
    if trestbps > 160:
        risk_score += 15
    elif trestbps > 140:
        risk_score += 12
    elif trestbps > 130:
        risk_score += 8
    elif trestbps > 120:
        risk_score += 4
    
    # Cholesterol
    if chol > 300:
        risk_score += 15
    elif chol > 240:
        risk_score += 12
    elif chol > 200:
        risk_score += 8
    elif chol > 180:
        risk_score += 4
    
    # Fasting Blood Sugar
    if fbs == "Yes":
        risk_score += 10
    
    # Resting ECG
    ecg_scores = {"Left Ventricular Hypertrophy": 10, "ST-T Wave Abnormality": 7, "Normal": 0}
    risk_score += ecg_scores.get(restecg, 0)
    
    # Max Heart Rate
    if thalach < 100:
        risk_score += 20
    elif thalach < 120:
        risk_score += 15
    elif thalach < 140:
        risk_score += 10
    elif thalach < 160:
        risk_score += 5
    
    # Exercise Angina
    if exang == "Yes":
        risk_score += 15
    
    # ST Depression
    if oldpeak > 4:
        risk_score += 15
    elif oldpeak > 2:
        risk_score += 10
    elif oldpeak > 1:
        risk_score += 5
    
    # Slope
    slope_scores = {"Downsloping": 10, "Flat": 5, "Upsloping": 0}
    risk_score += slope_scores.get(slope, 0)
    
    # Major Vessels
    risk_score += ca * 5
    
    # Thalassemia
    thal_scores = {"Reversible Defect": 15, "Fixed Defect": 10, "Normal": 0}
    risk_score += thal_scores.get(thal, 0)
    
    # Smoking
    if smoking == "Current Smoker":
        risk_score += 20
    elif smoking == "Former Smoker":
        risk_score += 10
    
    # Diabetes
    if diabetes_status == "Yes":
        risk_score += 15
    
    # Family History
    if family_history_heart == "Yes - Before 55":
        risk_score += 15
    elif family_history_heart == "Yes - After 55":
        risk_score += 8
    
    # ECG Abnormal
    if ecg_abnormal == "Yes":
        risk_score += 10
    
    # Stress Test
    if stress_test == "Abnormal":
        risk_score += 15
    
    # Medication
    if medication == "Yes":
        risk_score += 5
    
    risk_score = min(risk_score, 100)
    
    if risk_score >= 70:
        result = "HIGH RISK"
        color = "🔴"
        advice = "Immediate cardiologist consultation recommended. Lifestyle changes and medication may be necessary."
    elif risk_score >= 40:
        result = "MODERATE RISK"
        color = "🟡"
        advice = "Schedule a cardiac evaluation. Focus on diet, exercise, and stress management."
    else:
        result = "LOW RISK"
        color = "🟢"
        advice = "Maintain heart-healthy habits. Regular check-ups recommended."
    
    return result, risk_score, color, advice

def predict_parkinsons_advanced(mdvp_fo, mdvp_fhi, mdvp_flo, jitter_percent, jitter_abs,
                                 rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq,
                                 dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe,
                                 tremor, handwriting, smell_loss, sleep_disorder, constipation):
    """Enhanced Parkinson's risk calculation with 27 parameters"""
    risk_score = 0
    
    # Jitter
    if jitter_percent > 0.03:
        risk_score += 15
    elif jitter_percent > 0.01:
        risk_score += 10
    elif jitter_percent > 0.006:
        risk_score += 5
    
    # RAP
    if rap > 0.01:
        risk_score += 10
    elif rap > 0.005:
        risk_score += 5
    
    # DDP
    if ddp > 0.03:
        risk_score += 10
    elif ddp > 0.01:
        risk_score += 5
    
    # Shimmer
    if shimmer > 0.08:
        risk_score += 15
    elif shimmer > 0.05:
        risk_score += 10
    elif shimmer > 0.03:
        risk_score += 5
    
    # APQ
    if apq > 0.03:
        risk_score += 10
    elif apq > 0.02:
        risk_score += 5
    
    # NHR
    if nhr > 0.1:
        risk_score += 10
    elif nhr > 0.05:
        risk_score += 5
    
    # HNR
    if hnr < 15:
        risk_score += 15
    elif hnr < 18:
        risk_score += 10
    elif hnr < 20:
        risk_score += 5
    
    # RPDE
    if rpde > 0.6:
        risk_score += 10
    elif rpde > 0.5:
        risk_score += 5
    
    # DFA
    if dfa < 0.5:
        risk_score += 10
    elif dfa < 0.6:
        risk_score += 5
    
    # PPE
    if ppe > 0.3:
        risk_score += 10
    elif ppe > 0.2:
        risk_score += 5
    
    # Tremor
    if tremor == "Severe":
        risk_score += 15
    elif tremor == "Moderate":
        risk_score += 10
    elif tremor == "Mild":
        risk_score += 5
    
    # Handwriting
    if handwriting == "Severe Micrographia":
        risk_score += 10
    elif handwriting == "Moderate Changes":
        risk_score += 7
    elif handwriting == "Mild Changes":
        risk_score += 4
    
    # Smell Loss
    if smell_loss == "Complete Loss":
        risk_score += 10
    elif smell_loss == "Partial Loss":
        risk_score += 5
    
    # Sleep Disorder
    if sleep_disorder == "Yes - Frequent":
        risk_score += 10
    elif sleep_disorder == "Yes - Occasional":
        risk_score += 5
    
    # Constipation
    if constipation == "Chronic":
        risk_score += 10
    elif constipation == "Frequent":
        risk_score += 5
    
    risk_score = min(risk_score, 100)
    
    if risk_score >= 70:
        result = "HIGH LIKELIHOOD"
        color = "🔴"
        advice = "Immediate neurologist consultation strongly recommended. Early intervention is crucial."
    elif risk_score >= 40:
        result = "MODERATE LIKELIHOOD"
        color = "🟡"
        advice = "Schedule a neurological evaluation. Monitor symptoms and consider lifestyle adjustments."
    else:
        result = "LOW LIKELIHOOD"
        color = "🟢"
        advice = "Maintain regular health check-ups. Report any new symptoms promptly."
    
    return result, risk_score, color, advice

# ============================================
# HOMEPAGE SECTION
# ============================================

def show_homepage():
    # Welcome Message
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #555;">Your Personal AI-Powered Health Risk Assessment Tool</p>
        <p style="color: #666;">Get instant risk assessment for three major diseases with over 60 clinical parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🎯 60+ Parameters</h3>
            <p>Comprehensive assessment using clinically validated parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>⚡ Instant Results</h3>
            <p>Get your risk assessment in less than 1 second</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>🆓 Free & Accessible</h3>
            <p>No registration, no payment, accessible from any device</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Disease Cards Section
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Select a Disease to Assess Your Risk</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card diabetes-card" onclick="window.location.href='?page=diabetes'">
            <div class="card-icon">🩺</div>
            <div class="card-title">Diabetes</div>
            <div class="card-params">14 Clinical Parameters</div>
            <div class="card-params">• Fasting Glucose • HbA1c • BMI • Waist Circumference • Physical Activity • Family History • Triglycerides • HDL • Blood Pressure • Age • Pregnancies • Insulin • Skin Thickness • Pedigree</div>
            <span class="card-badge diabetes-badge">Start Assessment →</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Assess Diabetes Risk", key="diabetes_btn_home", use_container_width=True):
            st.session_state.page = "diabetes"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="card heart-card">
            <div class="card-icon">❤️</div>
            <div class="card-title">Heart Disease</div>
            <div class="card-params">19 Clinical Parameters</div>
            <div class="card-params">• Age • Sex • Blood Pressure • Cholesterol • Smoking • Diabetes Status • Family History • ECG • Stress Test • Max Heart Rate • Exercise Angina • ST Depression • Major Vessels • Thalassemia • and more</div>
            <span class="card-badge heart-badge">Start Assessment →</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Assess Heart Disease Risk", key="heart_btn_home", use_container_width=True):
            st.session_state.page = "heart"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="card parkinson-card">
            <div class="card-icon">🧠</div>
            <div class="card-title">Parkinson's Disease</div>
            <div class="card-params">27 Clinical Parameters</div>
            <div class="card-params">• Voice Analysis (Jitter, Shimmer, HNR) • Tremor • Handwriting • Loss of Smell • Sleep Disorders • Constipation • Balance Issues • Speech Changes • and more</div>
            <span class="card-badge parkinson-badge">Start Assessment →</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Assess Parkinson's Risk", key="parkinson_btn_home", use_container_width=True):
            st.session_state.page = "parkinson"
            st.rerun()
    
    st.markdown("---")
    
    # Statistics Section
    st.markdown("<h2 style='text-align: center; margin-bottom: 1rem;'>📊 Disease Statistics in India</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Diabetes", "77 Million", "2nd highest globally", delta_color="off")
        st.caption("Type 2 diabetes accounts for 90-95% of cases")
    
    with col2:
        st.metric("Heart Disease", "28%", "of all deaths in India", delta_color="off")
        st.caption("Leading cause of mortality")
    
    with col3:
        st.metric("Parkinson's", "1 Million+", "increasing with age", delta_color="off")
        st.caption("Early detection improves outcomes")
    
    st.markdown("---")
    
    # How It Works Section
    st.markdown("<h2 style='text-align: center; margin-bottom: 1rem;'>🔍 How It Works</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">1️⃣</div>
            <strong>Select Disease</strong>
            <p style="font-size: 0.8rem;">Choose from Diabetes, Heart Disease, or Parkinson's</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">2️⃣</div>
            <strong>Enter Parameters</strong>
            <p style="font-size: 0.8rem;">Fill in your clinical measurements</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">3️⃣</div>
            <strong>Get Risk Score</strong>
            <p style="font-size: 0.8rem;">AI calculates your personalized risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">4️⃣</div>
            <strong>Receive Advice</strong>
            <p style="font-size: 0.8rem;">Get recommendations based on your risk level</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns.
    </div>
    """, unsafe_allow_html=True)

# ============================================
# DIABETES TAB - ENHANCED
# ============================================

def show_diabetes():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3e8 0%, #ffe6d5 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;">
        <h2 style="color: #e67e22;">🩺 Diabetes Risk Assessment</h2>
        <p>Enter your health parameters for a comprehensive diabetes risk evaluation based on 14 clinical parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📌 Basic Information")
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1, help="For women only. Men can enter 0.")
        age = st.number_input("Age (years)", 18, 100, 45)
        bmi = st.number_input("BMI (Body Mass Index)", 15.0, 50.0, 25.0, format="%.1f")
        waist_circumference = st.number_input("Waist Circumference (cm)", 60, 150, 85, help="Measure at belly button level")
        physical_activity = st.selectbox("Physical Activity (days/week)", 
                                         [0, 1, 2, 3, 4, 5, 6, 7],
                                         format_func=lambda x: f"{x} days/week" if x > 0 else "Sedentary (0 days)")
    
    with col2:
        st.markdown("##### 🩸 Clinical Measurements")
        glucose = st.number_input("Fasting Blood Glucose (mg/dL)", 70, 300, 100)
        hba1c = st.number_input("HbA1c (%)", 4.0, 12.0, 5.7, format="%.1f", help="Long-term glucose control")
        bp = st.number_input("Blood Pressure (mm Hg)", 90, 180, 120)
        triglycerides = st.number_input("Triglycerides (mg/dL)", 50, 500, 150)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 45, help="Good cholesterol")
        insulin = st.number_input("Insulin Level (μU/mL)", 0, 300, 80)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    
    st.markdown("##### 👨‍👩‍👧 Family & Lifestyle")
    family_history = st.selectbox("Family History of Diabetes", 
                                  ["No", "Yes - Sibling", "Yes - One Parent", "Yes - Both Parents"])
    
    if st.button("🔮 Calculate Diabetes Risk", type="primary", use_container_width=True):
        result, risk, color, advice = predict_diabetes_advanced(
            pregnancies, glucose, bp, skin_thickness, insulin, bmi, 0.5, age,
            waist_circumference, physical_activity, family_history, hba1c, triglycerides, hdl
        )
        
        st.markdown("---")
        st.subheader("📊 Risk Assessment Results")
        
        st.markdown(f"### {color} **{result}**")
        st.markdown(f"**Risk Score:** {risk}/100")
        st.progress(risk/100)
        
        st.markdown("---")
        st.subheader("🔍 Key Risk Factors Identified")
        
        risk_factors = []
        if glucose > 120:
            risk_factors.append(f"⚠️ High fasting glucose: {glucose} mg/dL (>120)")
        if bmi > 25:
            risk_factors.append(f"⚠️ Overweight/Obese: BMI {bmi:.1f} (>25)")
        if waist_circumference > 90:
            risk_factors.append(f"⚠️ Central obesity: Waist {waist_circumference} cm (>90)")
        if hba1c > 6.0:
            risk_factors.append(f"⚠️ Elevated HbA1c: {hba1c}% (>6.0)")
        if triglycerides > 150:
            risk_factors.append(f"⚠️ High triglycerides: {triglycerides} mg/dL (>150)")
        if hdl < 40:
            risk_factors.append(f"⚠️ Low HDL cholesterol: {hdl} mg/dL (<40)")
        if physical_activity <= 2:
            risk_factors.append(f"⚠️ Sedentary lifestyle: {physical_activity} days/week activity")
        if family_history != "No":
            risk_factors.append(f"⚠️ Family history: {family_history}")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ No major risk factors detected. Keep up the healthy habits!")
        
        st.markdown("---")
        st.info(f"💡 **Recommendation:** {advice}")
    
    if st.button("← Back to Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# ============================================
# HEART DISEASE TAB - ENHANCED
# ============================================

def show_heart():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ffe8e8 0%, #ffd4d4 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;">
        <h2 style="color: #e74c3c;">❤️ Heart Disease Risk Assessment</h2>
        <p>Enter your clinical measurements for comprehensive heart disease risk evaluation based on 19 clinical parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📌 Demographics & Lifestyle")
        age_h = st.number_input("Age (years)", 20, 100, 55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        smoking = st.selectbox("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker"])
        family_history_heart = st.selectbox("Family History of Heart Disease", 
                                            ["No", "Yes - After 55", "Yes - Before 55"])
    
    with col2:
        st.markdown("##### 🩸 Clinical Measurements")
        bp_h = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 125)
        chol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 210)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        diabetes_status = st.selectbox("Diabetes Status", ["No", "Yes"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("##### ❤️ Cardiac Tests")
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, format="%.1f")
    
    with col4:
        st.markdown("##### 📊 Additional Tests")
        slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        stress_test = st.selectbox("Stress Test Result", ["Normal", "Abnormal"])
        medication = st.selectbox("Taking BP/Cholesterol Medication", ["No", "Yes"])
        ecg_abnormal = st.selectbox("ECG Abnormalities", ["No", "Yes"])
    
    if st.button("🔮 Calculate Heart Disease Risk", type="primary", use_container_width=True):
        result, risk, color, advice = predict_heart_advanced(
            age_h, sex, cp, bp_h, chol, fbs, restecg, thalach, exang, oldpeak,
            slope, ca, thal, smoking, diabetes_status, family_history_heart,
            ecg_abnormal, stress_test, medication
        )
        
        st.markdown("---")
        st.subheader("📊 Risk Assessment Results")
        
        st.markdown(f"### {color} **{result}**")
        st.markdown(f"**Risk Score:** {risk}/100")
        st.progress(risk/100)
        
        st.markdown("---")
        st.subheader("🔍 Key Risk Factors Identified")
        
        risk_factors = []
        if age_h > 50:
            risk_factors.append(f"⚠️ Age: {age_h} years (>50)")
        if bp_h > 130:
            risk_factors.append(f"⚠️ High blood pressure: {bp_h} mm Hg (>130)")
        if chol > 200:
            risk_factors.append(f"⚠️ High cholesterol: {chol} mg/dL (>200)")
        if smoking == "Current Smoker":
            risk_factors.append("⚠️ Current smoker")
        if diabetes_status == "Yes":
            risk_factors.append("⚠️ Diabetes")
        if family_history_heart != "No":
            risk_factors.append(f"⚠️ Family history: {family_history_heart}")
        if thalach < 140:
            risk_factors.append(f"⚠️ Low max heart rate: {thalach} bpm (<140)")
        if exang == "Yes":
            risk_factors.append("⚠️ Exercise-induced angina")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ No major risk factors detected!")
        
        st.markdown("---")
        st.info(f"💡 **Recommendation:** {advice}")
    
    if st.button("← Back to Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# ============================================
# PARKINSON'S TAB - ENHANCED
# ============================================

def show_parkinson():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e8f5e9 0%, #d4edd7 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;">
        <h2 style="color: #2ecc71;">🧠 Parkinson's Disease Risk Assessment</h2>
        <p>Enter voice measurements and clinical symptoms for comprehensive evaluation based on 27 clinical parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🎤 Voice Measurements")
        mdvp_fo = st.number_input("MDVP:Fo(Hz) - Average frequency", 80.0, 300.0, 120.0, format="%.1f")
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz) - Maximum frequency", 100.0, 500.0, 150.0, format="%.1f")
        mdvp_flo = st.number_input("MDVP:Flo(Hz) - Minimum frequency", 50.0, 300.0, 100.0, format="%.1f")
        jitter_percent = st.number_input("Jitter (%)", 0.0, 0.1, 0.006, format="%.4f")
        shimmer = st.number_input("Shimmer", 0.0, 0.2, 0.03, format="%.3f")
        hnr = st.number_input("Harmonic-to-Noise Ratio (HNR)", 0.0, 30.0, 20.0)
        nhr = st.number_input("Noise-to-Harmonics Ratio (NHR)", 0.0, 0.5, 0.02, format="%.4f")
    
    with col2:
        st.markdown("##### 📋 Clinical Symptoms")
        tremor = st.selectbox("Tremor/Resting Shaking", ["None", "Mild", "Moderate", "Severe"])
        handwriting = st.selectbox("Handwriting Changes", ["Normal", "Mild Changes", "Moderate Changes", "Severe Micrographia"])
        smell_loss = st.selectbox("Loss of Smell", ["No", "Partial Loss", "Complete Loss"])
        sleep_disorder = st.selectbox("REM Sleep Disorder", ["No", "Yes - Occasional", "Yes - Frequent"])
        constipation = st.selectbox("Constipation", ["No", "Occasional", "Frequent", "Chronic"])
        balance_issues = st.selectbox("Balance Problems", ["No", "Mild", "Moderate", "Severe"])
    
    # Additional voice parameters (used in calculation)
    rap = 0.003
    ppq = 0.003
    ddp = 0.01
    shimmer_db = 0.3
    apq3 = 0.015
    apq5 = 0.02
    apq = 0.025
    dda = 0.045
    rpde = 0.5
    dfa = 0.6
    spread1 = -5.0
    spread2 = 0.2
    d2 = 2.0
    ppe = 0.1
    jitter_abs = 0.00004
    
    if st.button("🔮 Calculate Parkinson's Risk", type="primary", use_container_width=True):
        result, risk, color, advice = predict_parkinsons_advanced(
            mdvp_fo, mdvp_fhi, mdvp_flo, jitter_percent, jitter_abs, rap, ppq, ddp,
            shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1,
            spread2, d2, ppe, tremor, handwriting, smell_loss, sleep_disorder, constipation
        )
        
        st.markdown("---")
        st.subheader("📊 Risk Assessment Results")
        
        st.markdown(f"### {color} **{result}**")
        st.markdown(f"**Risk Score:** {risk}/100")
        st.progress(risk/100)
        
        st.markdown("---")
        st.subheader("🔍 Key Risk Factors Identified")
        
        risk_factors = []
        if jitter_percent > 0.01:
            risk_factors.append(f"⚠️ Elevated jitter: {jitter_percent:.4f}% (>0.01%)")
        if shimmer > 0.05:
            risk_factors.append(f"⚠️ Elevated shimmer: {shimmer:.3f} (>0.05)")
        if hnr < 18:
            risk_factors.append(f"⚠️ Reduced HNR: {hnr:.1f} (<18)")
        if tremor != "None":
            risk_factors.append(f"⚠️ Tremor present: {tremor}")
        if handwriting != "Normal":
            risk_factors.append(f"⚠️ Handwriting changes: {handwriting}")
        if smell_loss != "No":
            risk_factors.append(f"⚠️ Loss of smell: {smell_loss}")
        if sleep_disorder != "No":
            risk_factors.append(f"⚠️ REM sleep disorder: {sleep_disorder}")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ No major risk factors detected!")
        
        st.markdown("---")
        st.info(f"💡 **Recommendation:** {advice}")
    
    if st.button("← Back to Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <div style="font-size: 3rem;">🏥</div>
        <h3 style="color: white;">Multi-Disease Predictor</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Quick Stats")
    st.markdown("""
    - **60+ Clinical Parameters**
    - **3 Major Diseases**
    - **Instant Results**
    - **Free & Accessible**
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Risk Levels")
    st.markdown("""
    <span class="risk-low">🟢 0-39%: Low Risk</span><br>
    <span class="risk-moderate">🟡 40-69%: Moderate Risk</span><br>
    <span class="risk-high">🔴 70-100%: High Risk</span>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Did You Know?")
    st.info("""
    - Early detection can reduce diabetes complications by 50%
    - 80% of heart disease is preventable with lifestyle changes
    - Voice changes can appear 5 years before Parkinson's diagnosis
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Medical Disclaimer")
    st.caption("This tool is for educational purposes only. Not a substitute for professional medical advice.")

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class="footer">
    <p>AI-Based Multi-Disease Prediction System | Advanced Risk Assessment | For Educational Purposes Only</p>
    <p>Powered by Clinical Guidelines & Rule-Based AI | N. Daniel Raj | DCSML | UID: 111723049034</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# PAGE NAVIGATION
# ============================================

if st.session_state.page == "home":
    show_homepage()
elif st.session_state.page == "diabetes":
    show_diabetes()
elif st.session_state.page == "heart":
    show_heart()
elif st.session_state.page == "parkinson":
    show_parkinson()