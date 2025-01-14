def say_hello():
    print("Hello from utils.py!")


def compile_model():
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def say_model():
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def say_hello2(name):
    print("Hello from utils.py!", name)              