# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# Step 1: Create the "moons" dataset
# Two interleaving half circles (non-linearly separable)
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Step 2: Create an SVM model with RBF kernel
model = SVC(kernel='rbf', gamma='scale', C=1.0)

# Step 3: Train the model
model.fit(X, y)

# Step 4: Define a function to plot the decision boundary
def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(8, 6))

    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # Create a meshgrid for background classification color
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predict class for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Draw decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.title("SVM with RBF Kernel (Classifying Moons Dataset)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

# Step 5: Plot the result
plot_decision_boundary(model, X, y)
