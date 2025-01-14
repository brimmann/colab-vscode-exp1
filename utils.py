def say_hello():
    print("Hello from utils.py!")


def compile_model():
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])