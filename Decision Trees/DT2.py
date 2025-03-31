# Import necessary libraries
# sklearn.tree for decision tree functionality
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
# pandas for data handling
import pandas as pd
# graphviz for tree visualization
import graphviz

# =============================================
# 1. CREATE THE DATASET
# =============================================
# Define a dictionary containing our pizza ordering scenarios
# Features:
# - Hungry: 1 = Yes, 0 = No
# - Raining: 1 = Yes, 0 = No 
# - Leftovers: 1 = Yes, 0 = No
# Target:
# - Order_Pizza: 1 = Order, 0 = Don't order
data = {
    'Hungry': [1, 1, 1, 1, 0, 0, 1, 1],  # Hunger status for each example
    'Raining': [1, 0, 0, 1, 0, 1, 0, 0],  # Weather status
    'Leftovers': [0, 1, 0, 1, 1, 0, 1, 0],  # Leftovers availability
    'Order_Pizza': [1, 0, 1, 1, 0, 0, 0, 1]  # Decision outcome
}

# Convert dictionary to pandas DataFrame for easier manipulation
df = pd.DataFrame(data)

# Separate features (X) from target variable (y)
# Features are all columns except Order_Pizza
X = df[['Hungry', 'Raining', 'Leftovers']]
# Target is just the Order_Pizza column
y = df['Order_Pizza']

# =============================================
# 2. TRAIN THE DECISION TREE MODEL
# =============================================
# Create decision tree classifier instance
# criterion='entropy' means we use information gain for splits
# max_depth=3 limits the tree depth to prevent overfitting
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Train the model on our data
# This is where the tree learns the decision rules
model.fit(X, y)

# =============================================
# 3. DISPLAY THE DECISION RULES
# =============================================
# Generate text representation of the decision tree
# feature_names specifies what to call each feature in the output
tree_rules = export_text(model, feature_names=list(X.columns))

# Print the decision rules
print("Decision Rules:\n", tree_rules)

# =============================================
# 4. MAKE A PREDICTION
# =============================================
# Create a test case: Hungry=Yes(1), Raining=No(0), Leftovers=Yes(1)
# We format it as a DataFrame to match training data structure
test_case = pd.DataFrame([[1, 0, 1]], columns=X.columns)

# Use the trained model to predict for our test case
prediction = model.predict(test_case)

# Print a friendly prediction result
print("\nPrediction:", "ORDER PIZZA! 🍕" if prediction[0] == 1 else "DON'T order ❌")

# =============================================
# 5. VISUALIZE THE DECISION TREE
# =============================================
# Try to create a visualization (requires Graphviz installed)
try:
    # Generate Graphviz dot data for visualization
    dot_data = export_graphviz(model, 
                             feature_names=X.columns,  # Use our feature names
                             class_names=['No Pizza', 'Pizza'],  # Class labels
                             filled=True,  # Use color filling
                             rounded=True)  # Rounded nodes
    
    # Create graph from dot data
    graph = graphviz.Source(dot_data)
    
    # Render and save the visualization as PNG
    # cleanup=True removes temporary files
    graph.render('pizza_decision_tree', format='png', cleanup=True)
    print("Tree visualization saved as pizza_decision_tree.png")

except Exception as e:
    # If Graphviz isn't installed, show error message
    print(f"Graphviz error: {e}\nInstall Graphviz to enable visualization.")
    print("As an alternative, you can use matplotlib to view the tree:")
    print("""
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))
    plot_tree(model, feature_names=X.columns, class_names=['No', 'Pizza'], filled=True)
    plt.show()
    """)