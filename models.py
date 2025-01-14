from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense



def get_model1():
    inputs = Input(shape=(28, 28), name='input_layer')
    x = Flatten(name='flatten_layer')(inputs)
    x = Dense(128, activation='relu', name='hidden_dense')(x)
    outputs = Dense(10, activation='softmax', name='output_layer')(x)
    return Model(inputs=inputs, outputs=outputs, name='functional_model')

def get_model2():
    inputs = Input(shape=(28, 28), name='input_layer')
    x = Flatten(name='flatten_layer')(inputs)
    x = Dense(128, activation='relu', name='hidden_dense')(x)
    outputs = Dense(10, activation='softmax', name='output_layer')(x)
    return Model(inputs=inputs, outputs=outputs, name='functional_model')

def compile_model(model):
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


