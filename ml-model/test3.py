'''
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)

img_height, img_width = 150, 150
channels = 3
batch_size = 32
epochs = 30

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data1a")

# Simple preprocessing function
def preprocess_func(img):
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Convert back to 3 channels
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Combine original and edges
    img = img.astype(float)
    edges_3ch = edges_3ch.astype(float)
    combined = img + edges_3ch * 0.3
    
    return np.clip(combined / 255.0, 0, 1)

# Data generators with custom preprocessing
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    preprocessing_function=lambda x: preprocess_func((x * 255).astype(np.uint8))
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: preprocess_func((x * 255).astype(np.uint8))
)

train_data = train_gen.flow_from_directory(
    os.path.join(data_dir, "training"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    os.path.join(data_dir, "validation"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# Build model
base_model = MobileNetV2(
    input_shape=(img_height, img_width, channels),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    verbose=1
)

# Save
save_path = os.path.join(script_dir, "mobilenetv2_damage_classifier.keras")
model.save(save_path)

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
'''
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)

img_height, img_width = 150, 150
channels = 3
batch_size = 32
epochs = 30

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data1a")

# Simple preprocessing function
def preprocess_func(img):
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Convert back to 3 channels
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Combine original and edges
    img = img.astype(float)
    edges_3ch = edges_3ch.astype(float)
    combined = img + edges_3ch * 0.3
    
    return np.clip(combined / 255.0, 0, 1)

# Data generators with custom preprocessing
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    preprocessing_function=lambda x: preprocess_func((x * 255).astype(np.uint8))
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: preprocess_func((x * 255).astype(np.uint8))
)

train_data = train_gen.flow_from_directory(
    os.path.join(data_dir, "training"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    os.path.join(data_dir, "validation"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# Build model
base_model = MobileNetV2(
    input_shape=(img_height, img_width, channels),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    verbose=1
)

# Save
save_path = os.path.join(script_dir, "mobilenetv2_damage_classifier.keras")
model.save(save_path)

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")



