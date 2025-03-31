from sklearn.linear_model import LinearRegression

# Sample data: study hours and exam scores
X = [[1], [2], [3], [4], [5]]  # Independent variable (e.g., study hours)
y = [2, 4, 5, 4, 5]  # Dependent variable (e.g., exam scores)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict a new value (e.g., for 6 study hours)
predicted_score = model.predict([[6]])
print(f"Predicted score for 6 study hours: {predicted_score[0]:.2f}")
