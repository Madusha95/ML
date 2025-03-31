import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------------------------------
# Step 1: Generate synthetic data
# ---------------------------------------------------

# Create 2D data with 3 cluster centers
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300,     # 300 data points
                       centers=3,         # 3 clusters
                       cluster_std=0.7,   # Cluster spread
                       random_state=42)   # For reproducibility

# ---------------------------------------------------
# Step 2: Create and fit the KMeans model
# ---------------------------------------------------

k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Predict cluster labels for each point
y_kmeans = kmeans.predict(X)

# ---------------------------------------------------
# Step 3: Visualize the clusters
# ---------------------------------------------------

# Plot data points with cluster labels (color-coded)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Data points')

# Plot the centroids found by KMeans
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

plt.title("K-Means Clustering (k=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
