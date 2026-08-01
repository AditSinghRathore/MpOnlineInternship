"""
Generates a synthetic heart.csv matching the schema of the Kaggle
'Heart Disease Dataset' by johnsmith88 (itself based on the UCI Cleveland set).

Columns:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal, target

If you have internet access, replace this file's output with the real
dataset downloaded directly from Kaggle for the actual submission.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1025  # matches the real dataset's row count

age = np.random.randint(29, 78, n_samples)
sex = np.random.randint(0, 2, n_samples)
cp = np.random.randint(0, 4, n_samples)
trestbps = np.random.randint(94, 201, n_samples)
chol = np.random.randint(126, 565, n_samples)
fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
restecg = np.random.randint(0, 3, n_samples)
thalach = np.random.randint(71, 203, n_samples)
exang = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])
oldpeak = np.round(np.random.uniform(0, 6.2, n_samples), 1)
slope = np.random.randint(0, 3, n_samples)
ca = np.random.randint(0, 5, n_samples)
thal = np.random.randint(0, 4, n_samples)

# Build target with a logical relationship to features (not pure noise)
risk_score = (
    (cp >= 2).astype(int) * 2
    + (thalach < 140).astype(int) * 2
    + (exang == 1).astype(int) * 2
    + (oldpeak > 2).astype(int) * 2
    + (ca >= 1).astype(int) * 2
    + (age > 55).astype(int)
    + (chol > 240).astype(int)
    + (sex == 1).astype(int)
    + np.random.normal(0, 2, n_samples)
)
target = (risk_score > np.median(risk_score)).astype(int)

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
    "target": target,
})

df.to_csv("heart.csv", index=False)
print(f"Generated heart.csv with {len(df)} rows")
print(df["target"].value_counts())
