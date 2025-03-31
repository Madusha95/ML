# Import libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dataset: [study_hours, attendance%, midterm_score, practice_tests]
X = np.array([
    [12, 95, 78, 5],  # Pass (1)
    [4, 60, 42, 2],   # Fail (0)
    [15, 100, 85, 7], # Pass (1)
    [3, 70, 38, 1],   # Fail (0)
    [10, 90, 72, 4],  # Pass (1)
    [5, 65, 45, 3]    # Fail (0)
])

# Labels: 0=Fail, 1=Pass
y = np.array([1, 0, 1, 0, 1, 0])

# Create Random Forest with 200 trees
model = RandomForestClassifier(n_estimators=200, 
                              random_state=42,
                              max_depth=3)  # Prevent overfitting

# Train the model
model.fit(X, y)

# Predict for new student: [8 hrs, 85%, 65 score, 3 tests]
new_student = np.array([[8, 85, 65, 3]])
prediction = model.predict(new_student)

# Print result with emoji
print("Prediction:", "Likely to pass 🎓" if prediction[0] == 1 else "Risk of failing ❌")

# Feature importance
features = ["Study Hours", "Attendance", "Midterm", "Practice Tests"]
importance = model.feature_importances_
print("\nWhat Matters Most:")
for feat, imp in zip(features, importance):
    print(f"• {feat}: {imp*100:.1f}%")

# Show probability
prob = model.predict_proba(new_student)[0]
print(f"\nPass probability: {prob[1]*100:.1f}%")

# Bonus: Decision path explanation (simplified)
print("\nKey Factors in Decision:")
if new_student[0][1] < 75:  # Attendance check
    print("- Low attendance warning!")
if new_student[0][2] < 50:  # Midterm check
    print("- Midterm score too low")