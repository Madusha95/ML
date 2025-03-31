# Import libraries - Note we use RandomForestRegressor now!
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

# Dataset: [sqft, beds, baths, year, neighborhood]
X = np.array([
    [1500, 3, 2, 1995, 7],  # $350,000
    [2200, 4, 3, 2010, 9],   # $550,000
    [1200, 2, 1, 1985, 5],   # $250,000
    [3000, 5, 4, 2020, 10],  # $750,000
    [1800, 3, 2, 2005, 6],   # $400,000
    [2400, 4, 3, 2015, 8]    # $600,000
])

# Home prices in dollars
y = np.array([350000, 550000, 250000, 750000, 400000, 600000])

# Create Random Forest Regressor
model = RandomForestRegressor(n_estimators=200,
                            random_state=42,
                            max_depth=4)  # Prevent overfitting

# Train the model
model.fit(X, y)

# Predict price for new house: [2000 sqft, 3 bed, 2.5 bath, 2012, 8]
new_house = np.array([[2000, 3, 2.5, 2012, 8]])
predicted_price = model.predict(new_house)

# Format output
formatted_price = "${:,.2f}".format(predicted_price[0])
print(f"Predicted home value: {formatted_price}")

# Feature importance
features = ["Square Feet", "Bedrooms", "Bathrooms", "Year", "Neighborhood"]
importance = pd.DataFrame({
    'Feature': features,
    'Impact (%)': model.feature_importances_ * 100
}).sort_values('Impact (%)', ascending=False)

print("\nWhat Impacts Value Most:")
print(importance.to_string(index=False))

# Show price range from different trees (confidence interval)
tree_predictions = [tree.predict(new_house) for tree in model.estimators_]
print(f"\nPrice Range Across Trees: ${min(tree_predictions)[0]/1000:.1f}K - ${max(tree_predictions)[0]/1000:.1f}K")