# Import necessary libraries
from sklearn import datasets  # For loading sample datasets
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.svm import SVC  # SVC = Support Vector Classification
from sklearn.metrics import accuracy_score  # For evaluating the model
import matplotlib.pyplot as plt  # For visualization
import numpy as np

# Load the iris dataset
iris = datasets.load_iris()

# For simplicity, we’ll take only two features (petal length and petal width)
X = iris.data[:, 2:4]  # Features: petal length and petal width
y = iris.target        # Labels: 0 = Setosa, 1 = Versicolor, 2 = Virginica

# To keep it binary, let’s filter only class 0 and 1 (Setosa and Versicolor)
X = X[y != 2]
y = y[y != 2]

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the SVM model with a linear kernel
model = SVC(kernel='linear')  # Try 'rbf' or 'poly' for nonlinear problems

# Train the model using the training data
model.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the decision boundary
def plot_decision_boundary(X, y, model):
    # Create a mesh grid of points to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predict for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('SVM Decision Boundary')
    plt.show()

# Call the visualization function
plot_decision_boundary(X, y, model)
