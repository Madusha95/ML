# Import libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dataset: [glucose, BMI, age, blood_pressure]
X = np.array([
    [148, 33.6, 50, 72],   # Diabetic (1)
    [85, 26.6, 31, 66],    # Healthy (0)
    [183, 23.3, 32, 64],   # Diabetic (1)
    [89, 28.1, 21, 76],    # Healthy (0)
    [137, 43.1, 33, 88],   # Diabetic (1)
    [116, 25.6, 30, 74]    # Healthy (0)
])

# Labels: 0=Healthy, 1=Diabetic
y = np.array([1, 0, 1, 0, 1, 0])

# Create Random Forest with 150 trees
model = RandomForestClassifier(n_estimators=150, random_state=42)

# Train the model
model.fit(X, y)

# Predict for new patient: [glucose=142, BMI=32, age=45, bp=80]
new_patient = np.array([[142, 32, 45, 80]])
prediction = model.predict(new_patient)

# Print result
print("Prediction:", "Diabetes risk 🚨" if prediction[0] == 1 else "Healthy ✅")

# Feature importance
features = ["Glucose", "BMI", "Age", "Blood Pressure"]
importance = model.feature_importances_
print("\nFeature Importance:")
for feat, imp in zip(features, importance):
    print(f"{feat}: {imp:.2f}")

# Bonus: Prediction probabilities
prob = model.predict_proba(new_patient)[0]
print(f"\nProbability: {prob[1]*100:.1f}% chance of diabetes")