import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST Dataset
mnist_data = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# Print first image pixel values
np.set_printoptions(linewidth=200)

print(train_images[0])
print("Label:", train_labels[0])

# Show image
plt.imshow(train_images[0], cmap='gray')

plt.title("Sample Digit")

plt.show()

# Normalize Images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images for CNN
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Build CNN Model
model = tf.keras.Sequential([

    tf.keras.Input(shape=(28,28,1)),

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        256,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        64,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )
])

# Compile Model
model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    epochs=5
)