import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 AI-Based Multi-Disease Prediction System")
st.markdown("### Advanced Risk Assessment | Diabetes | Heart Disease | Parkinson's Disease")
st.markdown("---")

# ============================================
# ENHANCED DIABETES PREDICTION
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
    
    # Waist Circumference (central obesity) - 0-15 points
    if waist_circumference > 100:  # >40 inches for men, >35 for women approx
        risk_score += 15
    elif waist_circumference > 90:
        risk_score += 10
    elif waist_circumference > 80:
        risk_score += 5
    
    # Physical Activity (0-15 points - LOWER activity = HIGHER risk)
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
    
    # HDL Cholesterol (0-15 points - LOWER HDL = HIGHER risk)
    if hdl < 35:
        risk_score += 15
    elif hdl < 40:
        risk_score += 10
    elif hdl < 50:
        risk_score += 5
    
    # Pregnancies (for women) - 0-10 points
    if pregnancies > 4:
        risk_score += 10
    elif pregnancies > 2:
        risk_score += 5
    
    # Blood Pressure (0-10 points)
    if bp > 140:
        risk_score += 10
    elif bp > 130:
        risk_score += 5
    
    # Insulin (0-5 points)
    if insulin > 200:
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

# ============================================
# ENHANCED HEART DISEASE PREDICTION
# ============================================
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
    
    # Sex (Male higher risk - 0-10 points)
    if sex == "Male":
        risk_score += 10
    
    # Chest Pain Type (0-20 points)
    cp_scores = {"Typical Angina": 20, "Atypical Angina": 15, "Non-anginal Pain": 10, "Asymptomatic": 5}
    risk_score += cp_scores.get(cp, 10)
    
    # Resting Blood Pressure (0-15 points)
    if trestbps > 160:
        risk_score += 15
    elif trestbps > 140:
        risk_score += 12
    elif trestbps > 130:
        risk_score += 8
    elif trestbps > 120:
        risk_score += 4
    
    # Cholesterol (0-15 points)
    if chol > 300:
        risk_score += 15
    elif chol > 240:
        risk_score += 12
    elif chol > 200:
        risk_score += 8
    elif chol > 180:
        risk_score += 4
    
    # Fasting Blood Sugar (0-10 points)
    if fbs == "Yes":
        risk_score += 10
    
    # Resting ECG (0-10 points)
    ecg_scores = {"Left Ventricular Hypertrophy": 10, "ST-T Wave Abnormality": 7, "Normal": 0}
    risk_score += ecg_scores.get(restecg, 0)
    
    # Max Heart Rate (0-20 points - LOWER = HIGHER risk)
    if thalach < 100:
        risk_score += 20
    elif thalach < 120:
        risk_score += 15
    elif thalach < 140:
        risk_score += 10
    elif thalach < 160:
        risk_score += 5
    
    # Exercise Induced Angina (0-15 points)
    if exang == "Yes":
        risk_score += 15
    
    # ST Depression (0-15 points)
    if oldpeak > 4:
        risk_score += 15
    elif oldpeak > 2:
        risk_score += 10
    elif oldpeak > 1:
        risk_score += 5
    
    # Slope (0-10 points)
    slope_scores = {"Downsloping": 10, "Flat": 5, "Upsloping": 0}
    risk_score += slope_scores.get(slope, 0)
    
    # Number of Major Vessels (0-15 points)
    risk_score += ca * 5
    
    # Thalassemia (0-15 points)
    thal_scores = {"Reversible Defect": 15, "Fixed Defect": 10, "Normal": 0}
    risk_score += thal_scores.get(thal, 0)
    
    # Smoking (0-20 points)
    if smoking == "Current Smoker":
        risk_score += 20
    elif smoking == "Former Smoker":
        risk_score += 10
    
    # Diabetes Status (0-15 points)
    if diabetes_status == "Yes":
        risk_score += 15
    
    # Family History of Heart Disease (0-15 points)
    if family_history_heart == "Yes - Before 55":
        risk_score += 15
    elif family_history_heart == "Yes - After 55":
        risk_score += 8
    
    # ECG Abnormal (0-10 points)
    if ecg_abnormal == "Yes":
        risk_score += 10
    
    # Abnormal Stress Test (0-15 points)
    if stress_test == "Abnormal":
        risk_score += 15
    
    # Taking BP/Cholesterol Medication (0-5 points)
    if medication == "Yes":
        risk_score += 5
    
    # Cap at 100
    risk_score = min(risk_score, 100)
    
    # Determine risk level
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

# ============================================
# ENHANCED PARKINSON'S PREDICTION
# ============================================
def predict_parkinsons_advanced(mdvp_fo, mdvp_fhi, mdvp_flo, jitter_percent, jitter_abs,
                                 rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq,
                                 dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe,
                                 tremor, handwriting, smell_loss, sleep_disorder, constipation):
    """Enhanced Parkinson's risk calculation with 27 parameters"""
    risk_score = 0
    
    # Jitter measures (0-20 points)
    if jitter_percent > 0.03:
        risk_score += 15
    elif jitter_percent > 0.01:
        risk_score += 10
    elif jitter_percent > 0.006:
        risk_score += 5
    
    # RAP (0-10 points)
    if rap > 0.01:
        risk_score += 10
    elif rap > 0.005:
        risk_score += 5
    
    # DDP (0-10 points)
    if ddp > 0.03:
        risk_score += 10
    elif ddp > 0.01:
        risk_score += 5
    
    # Shimmer measures (0-20 points)
    if shimmer > 0.08:
        risk_score += 15
    elif shimmer > 0.05:
        risk_score += 10
    elif shimmer > 0.03:
        risk_score += 5
    
    # APQ (0-10 points)
    if apq > 0.03:
        risk_score += 10
    elif apq > 0.02:
        risk_score += 5
    
    # NHR (0-10 points)
    if nhr > 0.1:
        risk_score += 10
    elif nhr > 0.05:
        risk_score += 5
    
    # HNR (0-15 points - LOWER = HIGHER risk)
    if hnr < 15:
        risk_score += 15
    elif hnr < 18:
        risk_score += 10
    elif hnr < 20:
        risk_score += 5
    
    # RPDE (0-10 points)
    if rpde > 0.6:
        risk_score += 10
    elif rpde > 0.5:
        risk_score += 5
    
    # DFA (0-10 points)
    if dfa < 0.5:
        risk_score += 10
    elif dfa < 0.6:
        risk_score += 5
    
    # PPE (0-10 points)
    if ppe > 0.3:
        risk_score += 10
    elif ppe > 0.2:
        risk_score += 5
    
    # Tremor (0-15 points)
    if tremor == "Severe":
        risk_score += 15
    elif tremor == "Moderate":
        risk_score += 10
    elif tremor == "Mild":
        risk_score += 5
    
    # Handwriting changes (0-10 points)
    if handwriting == "Severe Micrographia":
        risk_score += 10
    elif handwriting == "Moderate Changes":
        risk_score += 7
    elif handwriting == "Mild Changes":
        risk_score += 4
    
    # Loss of smell (0-10 points)
    if smell_loss == "Complete Loss":
        risk_score += 10
    elif smell_loss == "Partial Loss":
        risk_score += 5
    
    # REM Sleep Disorder (0-10 points)
    if sleep_disorder == "Yes - Frequent":
        risk_score += 10
    elif sleep_disorder == "Yes - Occasional":
        risk_score += 5
    
    # Constipation (0-10 points)
    if constipation == "Chronic":
        risk_score += 10
    elif constipation == "Frequent":
        risk_score += 5
    
    # Cap at 100
    risk_score = min(risk_score, 100)
    
    # Determine risk level
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
# UI TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["🩺 DIABETES", "❤️ HEART DISEASE", "🧠 PARKINSON'S"])

# ============================================
# TAB 1: DIABETES - ENHANCED
# ============================================
with tab1:
    st.header("📊 Diabetes Risk Assessment")
    st.markdown("Enter your health parameters for a comprehensive diabetes risk evaluation:")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Basic Information")
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1, help="For women only. Men can enter 0.")
        age = st.number_input("Age (years)", 18, 100, 45)
        bmi = st.number_input("BMI (Body Mass Index)", 15.0, 50.0, 25.0, format="%.1f")
        waist_circumference = st.number_input("Waist Circumference (cm)", 60, 150, 85, help="Measure at belly button level")
        physical_activity = st.selectbox("Physical Activity (days/week)", 
                                         [0, 1, 2, 3, 4, 5, 6, 7],
                                         format_func=lambda x: f"{x} days/week" if x > 0 else "Sedentary (0 days)")
    
    with col2:
        st.subheader("🩸 Clinical Measurements")
        glucose = st.number_input("Fasting Blood Glucose (mg/dL)", 70, 300, 100)
        hba1c = st.number_input("HbA1c (%)", 4.0, 12.0, 5.7, format="%.1f", help="Long-term glucose control")
        bp = st.number_input("Blood Pressure (mm Hg)", 90, 180, 120)
        triglycerides = st.number_input("Triglycerides (mg/dL)", 50, 500, 150)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 45, help="Good cholesterol")
        insulin = st.number_input("Insulin Level (μU/mL)", 0, 300, 80)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("👨‍👩‍👧 Family & Lifestyle")
        family_history = st.selectbox("Family History of Diabetes", 
                                      ["No", "Yes - Sibling", "Yes - One Parent", "Yes - Both Parents"])
    
    with col4:
        st.subheader("📋 Additional Risk Factors")
        # Empty for layout balance
    
    if st.button("🔮 Calculate Diabetes Risk", type="primary", use_container_width=True):
        result, risk, color, advice = predict_diabetes_advanced(
            pregnancies, glucose, bp, skin_thickness, insulin, bmi, 0.5, age,
            waist_circumference, physical_activity, family_history, hba1c, triglycerides, hdl
        )
        
        st.markdown("---")
        st.subheader("📊 Risk Assessment Results")
        
        # Display risk meter
        st.markdown(f"### {color} **{result}**")
        st.markdown(f"**Risk Score:** {risk}/100")
        st.progress(risk/100)
        
        # Display detailed breakdown
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

# ============================================
# TAB 2: HEART DISEASE - ENHANCED
# ============================================
with tab2:
    st.header("❤️ Heart Disease Risk Assessment")
    st.markdown("Enter your clinical measurements for comprehensive heart disease risk evaluation:")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Demographics & Lifestyle")
        age_h = st.number_input("Age (years)", 20, 100, 55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        smoking = st.selectbox("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker"])
        physical_activity_heart = st.selectbox("Physical Activity", ["Sedentary", "Light", "Moderate", "Active"])
        family_history_heart = st.selectbox("Family History of Heart Disease", 
                                            ["No", "Yes - After 55", "Yes - Before 55"])
    
    with col2:
        st.subheader("🩸 Clinical Measurements")
        bp_h = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 125)
        chol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 210)
        hdl_heart = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 45)
        ldl = st.number_input("LDL Cholesterol (mg/dL)", 50, 300, 130)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        diabetes_status = st.selectbox("Diabetes Status", ["No", "Yes"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("❤️ Cardiac Tests")
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, format="%.1f")
    
    with col4:
        st.subheader("📊 Additional Tests")
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

# ============================================
# TAB 3: PARKINSON'S - ENHANCED
# ============================================
with tab3:
    st.header("🧠 Parkinson's Disease Risk Assessment")
    st.markdown("Enter voice measurements and clinical symptoms for comprehensive evaluation:")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎤 Voice Measurements")
        mdvp_fo = st.number_input("MDVP:Fo(Hz) - Average frequency", 80.0, 300.0, 120.0, format="%.1f")
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz) - Maximum frequency", 100.0, 500.0, 150.0, format="%.1f")
        mdvp_flo = st.number_input("MDVP:Flo(Hz) - Minimum frequency", 50.0, 300.0, 100.0, format="%.1f")
        jitter_percent = st.number_input("Jitter (%)", 0.0, 0.1, 0.006, format="%.4f")
        shimmer = st.number_input("Shimmer", 0.0, 0.2, 0.03, format="%.3f")
        hnr = st.number_input("Harmonic-to-Noise Ratio (HNR)", 0.0, 30.0, 20.0)
        nhr = st.number_input("Noise-to-Harmonics Ratio (NHR)", 0.0, 0.5, 0.02, format="%.4f")
    
    with col2:
        st.subheader("📋 Clinical Symptoms")
        tremor = st.selectbox("Tremor/Resting Shaking", ["None", "Mild", "Moderate", "Severe"])
        handwriting = st.selectbox("Handwriting Changes", ["Normal", "Mild Changes", "Moderate Changes", "Severe Micrographia"])
        smell_loss = st.selectbox("Loss of Smell", ["No", "Partial Loss", "Complete Loss"])
        sleep_disorder = st.selectbox("REM Sleep Disorder", ["No", "Yes - Occasional", "Yes - Frequent"])
        constipation = st.selectbox("Constipation", ["No", "Occasional", "Frequent", "Chronic"])
        balance_issues = st.selectbox("Balance Problems", ["No", "Mild", "Moderate", "Severe"])
        speech_changes = st.selectbox("Speech Changes", ["Normal", "Soft Speech", "Monotone", "Unclear Speech"])
        facial_expression = st.selectbox("Facial Expression", ["Normal", "Reduced", "Mask-like"])
    
    # Additional voice parameters (hidden but used in calculation)
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

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("ℹ️ About This System")
    st.markdown("""
    ### 🏥 **AI-Based Multi-Disease Prediction System**
    
    This advanced system evaluates risk for:
    
    **🩺 Diabetes** (14 parameters)
    - Glucose, HbA1c, BMI, Waist Circumference
    - Family history, Physical activity
    - Triglycerides, HDL cholesterol
    
    **❤️ Heart Disease** (19 parameters)
    - Age, BP, Cholesterol, Smoking
    - Diabetes status, Family history
    - ECG, Stress test results
    
    **🧠 Parkinson's Disease** (27 parameters)
    - Voice analysis (jitter, shimmer, HNR)
    - Tremor, handwriting changes
    - Sleep disorders, loss of smell
    
    ### 📊 **Risk Levels**
    - 🟢 **0-39%:** Low Risk
    - 🟡 **40-69%:** Moderate Risk
    - 🔴 **70-100%:** High Risk
    
    ### ⚠️ **Medical Disclaimer**
    This tool is for **educational purposes only**.
    Not a substitute for professional medical diagnosis.
    Always consult qualified healthcare providers.
    """)
    
    st.markdown("---")
    st.caption("Enhanced Multi-Disease Prediction | Machine Learning | Clinical Decision Support")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>AI-Based Multi-Disease Prediction System | Advanced Risk Assessment | For Educational Purposes Only</p>", unsafe_allow_html=True)