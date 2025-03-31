# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Step 1: Create simple 2D data
# Class 0 (like apples)
X0 = np.random.randn(20, 2) - 2  # Cluster near (-2, -2)
y0 = np.zeros(20)                # Label 0

# Class 1 (like oranges)
X1 = np.random.randn(20, 2) + 2  # Cluster near (+2, +2)
y1 = np.ones(20)                 # Label 1

# Combine both classes into one dataset
X = np.vstack((X0, X1))          # Combine features
y = np.hstack((y0, y1))          # Combine labels

# Step 2: Create the SVM model
model = SVC(kernel='linear')  # Linear kernel = straight line decision boundary

# Step 3: Train the model
model.fit(X, y)

# Step 4: Plot the decision boundary
def plot_svm(model, X, y):
    plt.figure(figsize=(8, 6))

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # Plot the decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    plt.contour(XX, YY, Z, colors='k',
                levels=[-1, 0, 1], alpha=0.7,
                linestyles=['--', '-', '--'])

    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.title("Simple SVM Example with Linear Kernel")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

# Step 5: Call the plot function
plot_svm(model, X, y)
