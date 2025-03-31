# Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Step 3: Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape to add a channel dimension (grayscale = 1 channel)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Class names for reference
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Step 4: Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Step 7: Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Step 8: Predict and visualize
i = np.random.randint(0, len(x_test))
prediction = model.predict(x_test[i:i+1])
predicted_label = class_names[np.argmax(prediction)]
actual_label = class_names[y_test[i]]

plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label} | Actual: {actual_label}")
plt.axis('off')
plt.show()
