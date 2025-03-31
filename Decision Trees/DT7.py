# Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# =============================================
# 1. CREATE BANK LOAN DATASET
# =============================================
# Sample data of loan applicants
data = {
    # Numerical features
    'Age': [25, 35, 45, 30, 50, 23, 40, 65],
    'Income': [30000, 45000, 80000, 55000, 120000, 28000, 75000, 90000],
    'Credit_Score': [650, 720, 800, 690, 850, 620, 780, 710],
    
    # Categorical features
    'Employment_Status': ['Unemployed', 'Employed', 'Self-Employed', 'Employed', 
                         'Self-Employed', 'Employed', 'Self-Employed', 'Retired'],
    'Loan_Amount': [5000, 15000, 30000, 20000, 50000, 8000, 35000, 25000],
    
    # Target variable
    'Loan_Approved': ['No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# =============================================
# 2. PREPROCESS THE DATA
# =============================================
# Initialize LabelEncoder for categorical columns
le = LabelEncoder()

# Convert categorical columns to numerical values
df['Employment_Status'] = le.fit_transform(df['Employment_Status']) 
# After conversion:
# Employed=1, Retired=2, Self-Employed=3, Unemployed=0

# Convert target variable
df['Loan_Approved'] = le.fit_transform(df['Loan_Approved'])
# After conversion: No=0, Yes=1

# =============================================
# 3. PREPARE TRAINING DATA
# =============================================
# Features (X) - All columns except target
X = df.drop('Loan_Approved', axis=1)

# Target (y) - Only the approval status
y = df['Loan_Approved']

# =============================================
# 4. CREATE AND TRAIN DECISION TREE MODEL
# =============================================
# Initialize model with parameters:
# - max_depth=3: Limits tree complexity
# - min_samples_split=2: Minimum samples to split a node
model = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
model.fit(X, y)  # Train the model

# =============================================
# 5. VISUALIZE DECISION RULES
# =============================================
print("Decision Rules:")
print(export_text(model, 
                feature_names=list(X.columns),
                class_names=['Rejected', 'Approved']))

# =============================================
# 6. MAKE PREDICTIONS
# =============================================
# New applicant: Age=38, Income=60000, Credit_Score=700, 
# Employment_Status='Employed'(1), Loan_Amount=22000
new_applicant = [[38, 60000, 700, 1, 22000]]

# Get prediction (0=Reject, 1=Approve)
prediction = model.predict(new_applicant)
print(f"\nPrediction: {'APPROVED' if prediction[0]==1 else 'REJECTED'}")

# =============================================
# 7. FEATURE IMPORTANCE ANALYSIS
# =============================================
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.2f}")