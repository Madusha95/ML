# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset: [Weight (grams), Texture (0=smooth, 1=bumpy), Color (0=red, 1=orange)]
X = np.array([
    [150, 0, 0],  # Apple
    [170, 0, 0],  # Apple
    [130, 1, 1],  # Orange
    [120, 1, 1],  # Orange
    [160, 0, 0],  # Apple
    [140, 1, 1]   # Orange
])

# Labels: 0=Apple, 1=Orange
y = np.array([0, 0, 1, 1, 0, 1])

# Split data into training and testing sets (optional, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier with 100 trees (estimators)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make a prediction on a new fruit
new_fruit = np.array([[145, 1, 1]])  # Weight=145g, Bumpy, Orange
prediction = model.predict(new_fruit)

# Print the result
if prediction[0] == 0:
    print("Prediction: Apple 🍎")
else:
    print("Prediction: Orange 🍊")

# Check model accuracy (optional)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")