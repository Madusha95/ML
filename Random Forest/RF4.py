# Import libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dataset: [time_spent, clicks, age, income_level]
X = np.array([
    [5.2, 12, 28, 3],  # Bought (1)
    [1.1, 2, 45, 2],   # Didn't buy (0)
    [7.8, 18, 32, 4],  # Bought (1)
    [0.5, 1, 60, 1],   # Didn't buy (0)
    [6.5, 15, 25, 3],  # Bought (1)
    [2.3, 3, 50, 2]    # Didn't buy (0)
])

# Labels: 0=No purchase, 1=Purchase
y = np.array([1, 0, 1, 0, 1, 0])

# Create Random Forest with 200 trees
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the model
model.fit(X, y)

# Predict for new customer: [6.1 mins, 14 clicks, age 30, income 3]
new_customer = np.array([[6.1, 14, 30, 3]])
prediction = model.predict(new_customer)

# Print result
print("Prediction:", "Will buy 🛒" if prediction[0] == 1 else "Won't buy ❌")

# Feature importance
features = ["Time spent", "Clicks", "Age", "Income"]
importance = model.feature_importances_
for feat, imp in zip(features, importance):
    print(f"{feat}: {imp:.2f}")

# Bonus: Predict probability
prob = model.predict_proba(new_customer)[0]
print(f"\nBuy probability: {prob[1]*100:.1f}%")