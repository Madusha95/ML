import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data: car speeds (mph) and stopping distances (feet)
X = np.array([ [4], [7], [8], [10], [12], [15], [18], [22], [24] ])  # Speeds
y = np.array([2, 6, 9, 13, 18, 26, 36, 60, 65])  # Stopping distances

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict stopping distance for a car speed of 20 mph
predicted_distance = model.predict([[20]])
print(f"Predicted stopping distance for 20 mph: {predicted_distance[0]:.2f} feet")
