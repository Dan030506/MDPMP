import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("="*60)
print("TRAINING DISEASE PREDICTION MODELS")
print("="*60)

# Create directories
os.makedirs('models/diabetes', exist_ok=True)
os.makedirs('models/heart', exist_ok=True)
os.makedirs('models/parkinsons', exist_ok=True)

# ============================================
# 1. DIABETES MODEL
# ============================================
print("\n🔵 1. Diabetes Model")
print("-"*40)

try:
    df = pd.read_csv('dataset/diabetes/diabetes.csv')
    print(f"Loaded: {df.shape}")
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    with open('models/diabetes/diabetes_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/diabetes/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/diabetes/features.pkl', 'wb') as f:
        pickle.dump(list(df.columns[:-1]), f)
    
    acc = model.score(X_scaled, y)
    print(f"✓ Saved! Accuracy: {acc*100:.2f}%")
    
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================
# 2. HEART MODEL
# ============================================
print("\n❤️ 2. Heart Disease Model")
print("-"*40)

try:
    df = pd.read_csv('dataset/heart/heart.csv')
    print(f"Loaded: {df.shape}")
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    with open('models/heart/heart_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/heart/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/heart/features.pkl', 'wb') as f:
        pickle.dump(list(df.columns[:-1]), f)
    
    acc = model.score(X_scaled, y)
    print(f"✓ Saved! Accuracy: {acc*100:.2f}%")
    
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================
# 3. PARKINSON'S MODEL
# ============================================
print("\n🟢 3. Parkinson's Model")
print("-"*40)

try:
    df = pd.read_csv('dataset/parkinsons/parkinsons.csv')
    print(f"Loaded: {df.shape}")
    
    X = df.drop('status', axis=1).values
    y = df['status'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    with open('models/parkinsons/parkinsons_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/parkinsons/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/parkinsons/features.pkl', 'wb') as f:
        pickle.dump(list(df.drop('status', axis=1).columns), f)
    
    acc = model.score(X_scaled, y)
    print(f"✓ Saved! Accuracy: {acc*100:.2f}%")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)

# Verify
print("\nVerifying saved models:")
for disease in ['diabetes', 'heart', 'parkinsons']:
    path = f'models/{disease}/{disease}_model.pkl'
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"✓ {disease.capitalize()}: {size:,} bytes")
    else:
        print(f"✗ {disease.capitalize()}: Missing")
        