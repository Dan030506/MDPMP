import pandas as pd
import numpy as np

print("Creating datasets...")

# 1. DIABETES DATASET (PIMA format)
np.random.seed(42)
n_diab = 768
diabetes_data = {
    'Pregnancies': np.random.randint(0, 17, n_diab),
    'Glucose': np.random.randint(50, 200, n_diab),
    'BloodPressure': np.random.randint(60, 140, n_diab),
    'SkinThickness': np.random.randint(0, 100, n_diab),
    'Insulin': np.random.randint(0, 900, n_diab),
    'BMI': np.random.uniform(15, 45, n_diab),
    'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.5, n_diab),
    'Age': np.random.randint(21, 81, n_diab),
}
df_diab = pd.DataFrame(diabetes_data)
# Create realistic target (diabetes risk increases with glucose, BMI, age)
df_diab['Outcome'] = ((df_diab['Glucose'] > 120) & (df_diab['BMI'] > 25) & (df_diab['Age'] > 30)).astype(int)
df_diab.to_csv('dataset/diabetes/diabetes.csv', index=False)
print(f"✓ Diabetes dataset: {df_diab.shape}")

# 2. HEART DATASET
n_heart = 1000
heart_data = {
    'age': np.random.randint(29, 77, n_heart),
    'sex': np.random.randint(0, 2, n_heart),
    'cp': np.random.randint(0, 4, n_heart),
    'trestbps': np.random.randint(94, 200, n_heart),
    'chol': np.random.randint(126, 564, n_heart),
    'fbs': np.random.randint(0, 2, n_heart),
    'restecg': np.random.randint(0, 3, n_heart),
    'thalach': np.random.randint(71, 202, n_heart),
    'exang': np.random.randint(0, 2, n_heart),
    'oldpeak': np.random.uniform(0, 6.2, n_heart),
    'slope': np.random.randint(0, 3, n_heart),
    'ca': np.random.randint(0, 4, n_heart),
    'thal': np.random.randint(0, 3, n_heart),
}
df_heart = pd.DataFrame(heart_data)
# Create realistic target
df_heart['target'] = ((df_heart['age'] > 50) & (df_heart['chol'] > 200) & (df_heart['thalach'] < 150)).astype(int)
df_heart.to_csv('dataset/heart/heart.csv', index=False)
print(f"✓ Heart dataset: {df_heart.shape}")

# 3. PARKINSON'S DATASET
n_park = 500
park_features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
    'spread2', 'D2', 'PPE'
]
park_data = {}
for feat in park_features:
    park_data[feat] = np.random.randn(n_park)
df_park = pd.DataFrame(park_data)
# Create realistic target
df_park['status'] = ((df_park['MDVP:Jitter(%)'] > 0.5) | (df_park['MDVP:Shimmer'] > 0.5)).astype(int)
df_park.to_csv('dataset/parkinsons/parkinsons.csv', index=False)
print(f"✓ Parkinson's dataset: {df_park.shape}")

print("\n✅ All datasets created successfully!")
