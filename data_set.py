import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the dataset
def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
