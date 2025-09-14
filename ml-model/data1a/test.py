import tensorflow as tf
from tensorflow.keras.datasets import mnist # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, Flatten # pyright: ignore[reportMissingImports]

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0   # normalize (0–1)

# 2. Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),        # flatten image
    Dense(128, activation='relu'),        # hidden layer
    Dense(10, activation='softmax')       # output (10 digits)
])

# 3. Compile model
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# 4. Train
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5. Evaluate
model.evaluate(x_test, y_test)
