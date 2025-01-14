import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Explore the data
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")

# Visualize the first image
plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0]}")
plt.show()