# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Sample dataset - real-world would have 1000s of rows
data = {
    'MonthlyCharges': [70, 90, 50, 100, 60, 85],
    'Contract': ['Monthly', 'Yearly', 'Monthly', 'Two-year', 'Yearly', 'Monthly'],
    'Tenure': [12, 24, 3, 36, 9, 6],
    'InternetService': ['DSL', 'Fiber', 'None', 'Fiber', 'DSL', 'Fiber'],
    'ServiceCalls': [2, 1, 5, 0, 3, 4],
    'Churn': [0, 0, 1, 0, 1, 1]  # 0=Stay, 1=Leave
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical features
le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])  # Monthly=0, Yearly=1, Two-year=2
df['InternetService'] = le.fit_transform(df['InternetService'])  # DSL=0, Fiber=1, None=2

# Prepare features and target
X = df.drop('Churn', axis=1).values
y = df['Churn'].values

# Create balanced Random Forest (handles class imbalance)
model = RandomForestClassifier(n_estimators=150,
                             class_weight='balanced',  # Adjusts for fewer churn cases
                             random_state=42,
                             oob_score=True)  # Out-of-bag evaluation

# Train
model.fit(X, y)

# Predict for new customer: [$85 monthly, Monthly contract, 5 months tenure, Fiber, 3 calls]
new_customer = np.array([[85, 0, 5, 1, 3]])
prediction = model.predict(new_customer)

# Human-readable output
services = {0: 'DSL', 1: 'Fiber', 2: 'None'}
contracts = {0: 'Monthly', 1: 'Yearly', 2: 'Two-year'}
print(f"Customer Profile:\n"
      f"- Monthly: ${new_customer[0][0]}\n"
      f"- Contract: {contracts[new_customer[0][1]]}\n"
      f"- Tenure: {new_customer[0][2]} months\n"
      f"- Internet: {services[new_customer[0][3]]}\n"
      f"- Service Calls: {new_customer[0][4]}")

print("\nPrediction:", "High churn risk 🔴" if prediction[0] == 1 else "Likely to stay 🟢")

# Feature importance
features = ['Monthly Charge', 'Contract', 'Tenure', 'Internet', 'Service Calls']
importance = pd.DataFrame({
    'Factor': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Churn Drivers:")
print(importance.to_string(index=False))

# Show probability and OOB score
prob = model.predict_proba(new_customer)[0]
print(f"\nChurn Probability: {prob[1]*100:.1f}%")
print(f"Model Confidence (OOB Score): {model.oob_score_:.1%}")