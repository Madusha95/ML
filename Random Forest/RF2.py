# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample dataset: [Income ($), Credit Score, Debt-to-Income Ratio (%)]
X = np.array([
    [75000, 720, 15],  # Approved (1)
    [45000, 680, 25],  # Denied (0)
    [90000, 780, 10],  # Approved (1)
    [60000, 650, 30],  # Denied (0)
    [80000, 700, 20],  # Approved (1)
    [50000, 600, 35]   # Denied (0)
])

# Labels: 0=Denied, 1=Approved
y = np.array([1, 0, 1, 0, 1, 0])

# Create a Random Forest model with 50 trees
model = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the model
model.fit(X, y)

# Predict a new applicant: [$65000 income, 710 credit score, 18% DTI]
new_applicant = np.array([[65000, 710, 18]])
prediction = model.predict(new_applicant)

# Print the result
if prediction[0] == 1:
    print("Prediction: Approved ✅")
else:
    print("Prediction: Denied ❌")

# Feature importance (which factors matter most?)
importance = model.feature_importances_
features = ["Income", "Credit Score", "DTI"]
for feature, imp in zip(features, importance):
    print(f"{feature}: {imp:.2f}")