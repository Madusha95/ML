# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Step 1: Create circular data (non-linearly separable)
# Two concentric circles
X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)

# Step 2: Create SVM model with RBF kernel (non-linear)
model = SVC(kernel='rbf', gamma='scale')  # RBF = Radial Basis Function

# Step 3: Train the model
model.fit(X, y)

# Step 4: Visualize the decision boundary
def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(8, 6))

    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # Create a meshgrid for background classification color
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predict over the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.title("SVM with RBF Kernel (Classifying Circles)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

# Step 5: Call the plot function
plot_decision_boundary(model, X, y)
