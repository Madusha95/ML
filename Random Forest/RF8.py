# Import libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd  # Just for nice display

# Dataset: [satisfaction, salary, years_at_company, wfh_days]
X = np.array([
    [8, 6500, 3, 10],   # Stayed (0)
    [3, 4800, 1, 2],    # Left (1)
    [9, 7200, 5, 15],   # Stayed (0)
    [2, 4200, 1, 0],    # Left (1)
    [7, 5800, 4, 8],    # Stayed (0)
    [4, 5100, 2, 3]     # Left (1)
])

# Labels: 0=Stay, 1=Leave
y = np.array([0, 1, 0, 1, 0, 1])

# Create Random Forest with feature names
model = RandomForestClassifier(n_estimators=100, 
                             random_state=42,
                             max_features='sqrt')  # Better for smaller datasets

# Train the model
model.fit(X, y)

# Predict for new employee: [6 satisfaction, $5300, 2 years, 5 WFH days]
new_employee = np.array([[6, 5300, 2, 5]])
prediction = model.predict(new_employee)

# Print human-readable result
action = "Likely to STAY 👍" if prediction[0] == 0 else "Likely to LEAVE 👋"
print(f"Prediction: {action}")

# Feature importance with pandas for clean display
features = ["Satisfaction", "Salary", "Tenure", "WFH Days"]
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance.to_string(index=False))

# Show probability and risk factors
prob = model.predict_proba(new_employee)[0]
print(f"\nRetention probability: {prob[0]*100:.1f}%")

# Risk assessment
if new_employee[0][0] < 5:  # Satisfaction check
    print("⚠️ Warning: Low job satisfaction!")
if new_employee[0][1] < 5000:  # Salary check
    print("⚠️ Warning: Below average salary!")