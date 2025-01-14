from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

# Define the input layer
inputs = Input(shape=(28, 28), name='input_layer')

# Flatten the input
x = Flatten(name='flatten_layer')(inputs)

# Hidden dense layer with 128 neurons and ReLU activation
x = Dense(128, activation='relu', name='hidden_dense')(x)

# Output dense layer with 10 neurons and softmax activation for classification
outputs = Dense(10, activation='softmax', name='output_layer')(x)

# Create the model by specifying the inputs and outputs
model1 = Model(inputs=inputs, outputs=outputs, name='functional_model')


