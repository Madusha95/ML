# Import required libraries
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# =============================================
# 1. CREATE THE PLANT WATERING DATASET
# =============================================
# Create a dictionary of plant watering scenarios
# Each key is a feature or the target decision
# Each value is a list of observations
data = {
    # Soil moisture levels (categorical)
    'Soil_Moisture': ['Dry', 'Dry', 'Moist', 'Wet', 'Dry', 'Moist', 'Moist', 'Wet'],
    
    # Days since last watering (numerical)
    'Days_Since_Watering': [3, 5, 2, 1, 4, 3, 1, 2],
    
    # Weather forecast (categorical)
    'Weather_Forecast': ['Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy', 'Sunny', 'Rainy'],
    
    # Target variable - whether we should water (Yes/No)
    'Water_Plant': ['Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No']
}

# Convert dictionary to pandas DataFrame
df = pd.DataFrame(data)

# =============================================
# 2. PREPROCESS THE DATA
# =============================================
# Initialize LabelEncoder for categorical features
le = LabelEncoder()

# Convert Soil_Moisture text to numbers
# Dry=0, Moist=1, Wet=2 (mapping may vary)
df['Soil_Moisture'] = le.fit_transform(df['Soil_Moisture'])

# Convert Weather_Forecast text to numbers
# Sunny=2, Rainy=1, Cloudy=0 (mapping may vary)
df['Weather_Forecast'] = le.fit_transform(df['Weather_Forecast'])

# Days_Since_Watering is already numerical - no encoding needed

# =============================================
# 3. PREPARE TRAINING DATA
# =============================================
# Select features (predictor variables)
X = df[['Soil_Moisture', 'Days_Since_Watering', 'Weather_Forecast']]

# Select target variable (what we want to predict)
y = df['Water_Plant']

# =============================================
# 4. CREATE AND TRAIN DECISION TREE MODEL
# =============================================
# Initialize Decision Tree Classifier
# criterion='gini' - uses Gini impurity for splits
# max_depth=3 - limits tree depth to prevent overfitting
model = DecisionTreeClassifier(criterion='gini', max_depth=3)

# Train the model on our data
model.fit(X, y)

# =============================================
# 5. VISUALIZE DECISION RULES
# =============================================
# Print the decision rules in text format
print("Decision Rules:")
print(export_text(model,
                feature_names=['Soil_Moisture', 'Days_Since_Watering', 'Weather_Forecast'],
                class_names=['No', 'Yes']))

# =============================================
# 6. MAKE A PREDICTION
# =============================================
# Test case: Soil=Dry(0), Days=3, Forecast=Sunny(2)
test_case = [[0, 3, 2]]

# Get prediction from model
prediction = model.predict(test_case)

# Print human-readable result
print("\nPrediction for [Dry, 3 days, Sunny]:", 
      "WATER PLANT! 💧" if prediction[0] == 'Yes' else "DON'T WATER ❌")

# =============================================
# 7. ANALYZE FEATURE IMPORTANCE
# =============================================
# Show which features most influenced decisions
print("\nFeature Importance:")
# Create dictionary of feature names and their importance scores
print(dict(zip(X.columns, model.feature_importances_)))