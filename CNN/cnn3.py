# Import TensorFlow and other essentials
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a fake grayscale image dataset (10x10 images, 2 classes)
num_samples = 1000
image_size = (10, 10, 1)  # 10x10 grayscale

# Generate random data
x_data = np.random.rand(num_samples, *image_size).astype("float32")

# Labels: class 0 if pixel sum < 50%, class 1 if pixel sum >= 50%
y_data = np.array([0 if img.sum() < 50 else 1 for img in x_data])

# Step 2: Split into train and test
split_index = int(0.8 * num_samples)
x_train, x_test = x_data[:split_index], x_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]

# Step 3: Build a simple CNN
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(10, 10, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='softmax')  # Output: 2 classes
])

# Step 4: Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Step 6: Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Step 7: Visualize a random image and its prediction
i = np.random.randint(0, len(x_test))
plt.imshow(x_test[i].reshape(10, 10), cmap='gray')
plt.title(f"Predicted: {np.argmax(model.predict(x_test[i:i+1]))}, Actual: {y_test[i]}")
plt.axis('off')
plt.show()
