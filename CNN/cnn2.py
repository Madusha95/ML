# Import necessary modules
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Step 1: Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Step 2: Normalize the images (scale pixel values to [0, 1])
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Step 3: Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # 32 filters
    layers.MaxPooling2D((2, 2)),  # Downsample
    layers.Conv2D(64, (3, 3), activation='relu'),  # More filters
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),  # Flatten for fully connected layers
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Step 7: Visualize a prediction
import numpy as np
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Pick a random test image
i = np.random.randint(0, len(x_test))
prediction = model.predict(x_test[i:i+1])
predicted_class = class_names[np.argmax(prediction)]

plt.imshow(x_test[i])
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()
