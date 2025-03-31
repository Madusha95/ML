from sklearn.tree import DecisionTreeClassifier

# Sample data: [Price, Weight] → Label (0 = Don't buy, 1 = Buy)
X = [[100, 1], [150, 2], [200, 3], [50, 0.5]]  
y = [1, 1, 0, 0]  # 1 = Buy, 0 = Don't buy

# Train the tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict: Should I buy a $120, 1.5kg item?
prediction = model.predict([[120, 1.5]])  
print("Buy?" , "Yes!" if prediction[0] == 1 else "No!")