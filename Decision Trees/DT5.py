# Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =============================================
# 1. LOAD DATASET FROM EXCEL FILE
# =============================================
# Read the Excel file into a pandas DataFrame
# Make sure the Excel file is in the same directory as your script
# or provide the full file path
try:
    # 'plant_data.xlsx' should have columns matching our features and target
    # Sheet_name specifies which sheet to read (0 = first sheet)
    df = pd.read_excel('plant_data.xlsx', sheet_name=0)
    
    # Display first 5 rows to verify loading
    print("Loaded data preview:")
    print(df.head())
    
except FileNotFoundError:
    print("Error: Excel file not found. Please check the file name and path.")
    exit()

# =============================================
# 2. PREPROCESS THE DATA
# =============================================
# Convert categorical columns to numerical values
# Using pandas' built-in get_dummies() for one-hot encoding
# This creates separate columns for each category
df_processed = pd.get_dummies(df, columns=['Soil_Moisture', 'Weather_Forecast'])

# Separate features (X) and target (y)
# Drop the target column to get features
X = df_processed.drop('Water_Plant', axis=1)
# Target is just the Water_Plant column
y = df_processed['Water_Plant']

# =============================================
# 3. SPLIT DATA INTO TRAINING AND TEST SETS
# =============================================
# Split data into 70% training, 30% testing
# random_state ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# =============================================
# 4. CREATE AND TRAIN DECISION TREE MODEL
# =============================================
# Initialize Decision Tree Classifier
# max_depth limits tree complexity to prevent overfitting
# min_samples_leaf requires this many samples in leaf nodes
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2)

# Train the model on training data
model.fit(X_train, y_train)

# =============================================
# 5. EVALUATE MODEL PERFORMANCE
# =============================================
# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# =============================================
# 6. VISUALIZE DECISION RULES
# =============================================
# Get feature names after one-hot encoding
feature_names = list(X.columns)

# Print text representation of decision rules
print("\nDecision Rules:")
print(export_text(model, 
                feature_names=feature_names,
                class_names=['No', 'Yes']))

# =============================================
# 7. MAKE A PREDICTION WITH NEW DATA
# =============================================
# Example: Predict for [Dry soil, 3 days since watering, Sunny forecast]
# Must format exactly like our training data after one-hot encoding
new_data = {
    'Days_Since_Watering': [3],
    'Soil_Moisture_Dry': [1],  # 1 = True
    'Soil_Moisture_Moist': [0],
    'Soil_Moisture_Wet': [0],
    'Weather_Forecast_Sunny': [1],
    'Weather_Forecast_Cloudy': [0],
    'Weather_Forecast_Rainy': [0]
}

# Convert to DataFrame with same columns as training data
new_df = pd.DataFrame(new_data, columns=feature_names)

# Make prediction
prediction = model.predict(new_df)
print("\nPrediction for new data:", 
      "WATER PLANT! 💧" if prediction[0] == 'Yes' else "DON'T WATER ❌")

# =============================================
# 8. SAVE MODEL FOR FUTURE USE (OPTIONAL)
# =============================================
# Uncomment to save the trained model to a file
# import joblib
# joblib.dump(model, 'plant_watering_model.joblib')

