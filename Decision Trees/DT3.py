# Import necessary libraries
# DecisionTreeClassifier for creating the decision tree model
# export_text for visualizing the tree rules in text format
from sklearn.tree import DecisionTreeClassifier, export_text

# LabelEncoder for converting categorical text data to numerical values
from sklearn.preprocessing import LabelEncoder

# pandas for data manipulation and analysis
import pandas as pd

# =============================================
# 1. CREATE THE DATASET
# =============================================
# Create a dictionary containing our running decision scenarios
# Each key represents a feature or the target variable
# Each value is a list of observations for that feature
data = {
    # Weather conditions for each observation
    'Weather': ['Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy', 'Cloudy', 'Sunny', 'Sunny'],
    
    # Temperature levels for each observation
    'Temperature': ['Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild', 'Hot'],
    
    # Humidity levels for each observation
    'Humidity': ['High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal'],
    
    # Target variable - whether we went running (Yes/No)
    'Go_Running': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

# Convert the dictionary to a pandas DataFrame for easier manipulation
df = pd.DataFrame(data)

# =============================================
# 2. PREPROCESS THE DATA
# =============================================
# Initialize the LabelEncoder to convert text categories to numerical values
le = LabelEncoder()

# Convert Weather text categories to numerical values
# Sunny=2, Rainy=1, Cloudy=0 (assignments may vary)
df['Weather'] = le.fit_transform(df['Weather'])

# Convert Temperature text categories to numerical values
# Hot=1, Mild=2, Cool=0 (assignments may vary)
df['Temperature'] = le.fit_transform(df['Temperature'])

# Convert Humidity text categories to numerical values
# High=0, Normal=1
df['Humidity'] = le.fit_transform(df['Humidity'])

# =============================================
# 3. PREPARE TRAINING DATA
# =============================================
# Select our features (predictor variables)
# We use all columns except the target variable
X = df[['Weather', 'Temperature', 'Humidity']]

# Select our target variable (what we want to predict)
y = df['Go_Running']

# =============================================
# 4. CREATE AND TRAIN THE DECISION TREE MODEL
# =============================================
# Initialize the Decision Tree Classifier
# criterion='entropy' means we use information gain for splits
# max_depth=3 limits how deep the tree can grow to prevent overfitting
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Train the model on our data
# This is where the tree learns the decision rules
model.fit(X, y)

# =============================================
# 5. VISUALIZE THE DECISION RULES
# =============================================
# Print the decision rules in a readable text format
print("Decision Rules:")
print(export_text(model, 
                feature_names=['Weather', 'Temperature', 'Humidity'],
                class_names=['No', 'Yes']))

# =============================================
# 6. MAKE A PREDICTION
# =============================================
# Create a test case for prediction:
# Weather: Sunny (encoded as 2)
# Temperature: Mild (encoded as 2)
# Humidity: Normal (encoded as 1)
test_case = [[2, 2, 1]]

# Use the trained model to make a prediction
prediction = model.predict(test_case)

# Print a human-readable prediction result
print("\nPrediction for [Sunny, Mild, Normal]:", 
      "GO RUN! 🏃‍♂️" if prediction[0] == 'Yes' else "DON'T GO ❌")

# =============================================
# 7. ANALYZE FEATURE IMPORTANCE
# =============================================
# Print which features were most important in making decisions
# Values range from 0 (not important) to 1 (very important)
print("\nFeature Importance:")
# Create a dictionary pairing feature names with their importance scores
print(dict(zip(X.columns, model.feature_importances_)))