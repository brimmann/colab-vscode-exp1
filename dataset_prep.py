import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

from types import SimpleNamespace

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

dataset = SimpleNamespace(train_images=train_images, test_images=test_images, train_labels=train_labels, test_labels=test_labels)