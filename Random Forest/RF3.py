# Import libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dataset: [count of "free", number of links, sender trust score (0-10)]
X = np.array([
    [5, 3, 2],  # Spam (1)
    [0, 1, 8],  # Not spam (0)
    [3, 4, 3],  # Spam (1)
    [1, 0, 9],  # Not spam (0)
    [4, 5, 1],  # Spam (1)
    [0, 1, 7]   # Not spam (0)
])

# Labels: 0=Not spam, 1=Spam
y = np.array([1, 0, 1, 0, 1, 0])

# Create Random Forest with 100 trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X, y)

# Predict a new email: [2 "free", 3 links, sender score=4]
new_email = np.array([[2, 3, 4]])
prediction = model.predict(new_email)

# Print result
print("Prediction:", "Spam 🚨" if prediction[0] == 1 else "Not spam ✅")

# Show which features matter most
features = ["'free' count", "Links", "Sender score"]
importance = model.feature_importances_
for feat, imp in zip(features, importance):
    print(f"{feat}: {imp:.2f}")