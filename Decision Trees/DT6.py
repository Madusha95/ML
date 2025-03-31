# Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# =============================================
# 1. CREATE SAMPLE STUDENT DATASET
# =============================================
# Create a dictionary of student data
data = {
    # Numerical features
    'Study_Hours': [2, 3, 5, 1, 4, 6, 2, 5],  
    'Practice_Test_Score': [65, 72, 88, 53, 79, 92, 61, 85],
    
    # Categorical features
    'Attended_Class': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Used_Resources': ['Textbook', 'Videos', 'Both', 'Neither', 'Both', 'Textbook', 'Neither', 'Videos'],
    
    # Target variable
    'Passed_Exam': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# =============================================
# 2. PREPROCESS THE DATA
# =============================================
# One-hot encode categorical columns
df_processed = pd.get_dummies(df, columns=['Attended_Class', 'Used_Resources'])

# Separate features (X) and target (y)
X = df_processed.drop('Passed_Exam', axis=1)  # All columns except target
y = df_processed['Passed_Exam']  # Only target column

# =============================================
# 3. TRAIN DECISION TREE MODEL
# =============================================
# Initialize model with parameters:
# - max_depth: Limits tree depth to prevent overfitting
# - min_samples_leaf: Minimum samples required in leaf nodes
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
model.fit(X, y)  # Train the model

# =============================================
# 4. VISUALIZE DECISION RULES
# =============================================
print("Decision Rules:")
print(export_text(model, 
                feature_names=list(X.columns),
                class_names=['Fail', 'Pass']))

# =============================================
# 5. MAKE PREDICTIONS
# =============================================
# Predict for new student:
# Study_Hours=4, Practice_Test_Score=82, Attended_Class=Yes, Used_Resources=Both
new_student = [[4, 82, 1, 0, 1, 0]]  # Encoded values

prediction = model.predict(new_student)
print(f"\nPrediction: {'PASS' if prediction[0]=='Yes' else 'FAIL'}")

# =============================================
# 6. FEATURE IMPORTANCE ANALYSIS
# =============================================
print("\nFeature Importances:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"{name}: {importance:.2f}")