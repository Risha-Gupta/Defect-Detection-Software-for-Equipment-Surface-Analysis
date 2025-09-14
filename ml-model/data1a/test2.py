# Import TensorFlow library for deep learning functionality
import tensorflow as tf

# Import Keras modules for building neural networks
from tensorflow import keras

# Import Sequential model class for building linear stack of layers
from tensorflow.keras.models import Sequential

# Import layer types needed for CNN architecture
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Import ImageDataGenerator for data preprocessing and augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import optimizers for training the model
from tensorflow.keras.optimizers import Adam

# Import OS module for file path operations
import os

# Print TensorFlow version to verify installation
print(f"TensorFlow version: {tf.__version__}")

# Set random seed for reproducible results
tf.random.set_seed(42)

# Define image dimensions and parameters
pic_height = 150  # Height in pixels for input images
pic_width = 150   # Width in pixels for input images
color_channels = 3  # Number of color channels (RGB)

# Define training parameters
batch_count = 32  # Number of images processed in each batch
training_rounds = 30  # Number of epochs to train the model
num_classes = 2  # Binary classification (2 classes)

# Create data generator for training images with augmentation
train_data_prep = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values from 0-255 to 0-1 range
    rotation_range=25,       # Randomly rotate images up to 25 degrees
    width_shift_range=0.2,   # Randomly shift images horizontally by 20%
    height_shift_range=0.2,  # Randomly shift images vertically by 20%
    shear_range=0.2,         # Apply random shear transformations
    zoom_range=0.2,          # Randomly zoom into images by up to 20%
    horizontal_flip=True,    # Randomly flip images horizontally
    brightness_range=[0.8, 1.2],  # Randomly adjust brightness
    fill_mode='nearest'      # Fill pixels after transformations using nearest neighbor
)

# Create data generator for validation images (no augmentation)
val_data_prep = ImageDataGenerator(
    rescale=1./255           # Only normalize pixel values, no augmentation for validation
)

# Load training images from directory structure
train_pics = train_data_prep.flow_from_directory(
    'training',                    # Directory containing training subdirectories
    target_size=(pic_height, pic_width),  # Resize all images to specified dimensions
    batch_size=batch_count,         # Number of images to load per batch
    class_mode='binary',           # Binary classification (0 or 1)
    shuffle=True                   # Randomly shuffle the training data
)

# Load validation images from directory structure
val_pics = val_data_prep.flow_from_directory(
    'validation',                  # Directory containing validation subdirectories
    target_size=(pic_height, pic_width),  # Resize all images to specified dimensions
    batch_size=batch_count,         # Number of images to load per batch
    class_mode='binary',           # Binary classification (0 or 1)
    shuffle=False                  # Don't shuffle validation data for consistent evaluation
)

# Print information about the loaded datasets
print(f"Training samples: {train_pics.samples}")      # Total number of training images
print(f"Validation samples: {val_pics.samples}")     # Total number of validation images
print(f"Class indices: {train_pics.class_indices}")   # Mapping of class names to indices


# Create a Sequential model (layers stacked one after another)
complex_model = Sequential()

# First convolutional block - detect basic features
complex_model.add(Conv2D(
    32,                           # Number of filters (feature detectors)
    (3, 3),                      # Filter size (3x3 kernel)
    activation='relu',           # ReLU activation function
    input_shape=(pic_height, pic_width, color_channels),  # Input image dimensions
    padding='same'               # Keep same spatial dimensions
))
complex_model.add(BatchNormalization())  # Normalize inputs to improve training stability
complex_model.add(MaxPooling2D(2, 2))    # Max pooling with 2x2 pool size to reduce dimensions

# Second convolutional block - detect more complex patterns
complex_model.add(Conv2D(
    64,                          # Increase number of filters to 64
    (3, 3),                      # 3x3 kernel size
    activation='relu',           # ReLU activation function
    padding='same'               # Keep same spatial dimensions
))
complex_model.add(BatchNormalization())  # Batch normalization for better training
complex_model.add(MaxPooling2D(2, 2))    # Max pooling to further reduce dimensions

# Third convolutional block - detect higher-level features
complex_model.add(Conv2D(
    128,                         # Increase number of filters to 128
    (3, 3),                      # 3x3 kernel size
    activation='relu',           # ReLU activation function
    padding='same'               # Keep same spatial dimensions
))
complex_model.add(BatchNormalization())  # Batch normalization layer
complex_model.add(MaxPooling2D(2, 2))    # Max pooling layer

# Fourth convolutional block - detect very complex patterns
complex_model.add(Conv2D(
    256,                         # Increase number of filters to 256
    (3, 3),                      # 3x3 kernel size
    activation='relu',           # ReLU activation function
    padding='same'               # Keep same spatial dimensions
))
complex_model.add(BatchNormalization())  # Batch normalization layer
complex_model.add(MaxPooling2D(2, 2))    # Max pooling layer

# Fifth convolutional block - detect most complex features
complex_model.add(Conv2D(
    512,                         # Increase number of filters to 512
    (3, 3),                      # 3x3 kernel size
    activation='relu',           # ReLU activation function
    padding='same'               # Keep same spatial dimensions
))
complex_model.add(BatchNormalization())  # Batch normalization layer
complex_model.add(MaxPooling2D(2, 2))    # Final max pooling layer

# Flatten the 2D feature maps into 1D vector for dense layers
complex_model.add(Flatten())

# Add dropout layer to prevent overfitting before classification layers
complex_model.add(Dropout(0.5))          # Randomly set 50% of inputs to 0 during training

# First fully connected (dense) layer for feature combination
complex_model.add(Dense(
    1024,                        # 1024 neurons in this layer
    activation='relu'            # ReLU activation function
))

# Add another dropout layer for additional regularization
complex_model.add(Dropout(0.3))          # Randomly set 30% of inputs to 0 during training

# Second fully connected layer for further feature processing
complex_model.add(Dense(
    512,                         # 512 neurons in this layer
    activation='relu'            # ReLU activation function
))

# Add final dropout layer before output
complex_model.add(Dropout(0.2))          # Randomly set 20% of inputs to 0 during training

# Final output layer for binary classification
complex_model.add(Dense(
    1,                           # Single neuron for binary output
    activation='sigmoid'         # Sigmoid activation for probability output (0-1)
))

# Compile the model with optimizer, loss function, and metrics
complex_model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Adam optimizer with lower learning rate for complex model
    loss='binary_crossentropy',            # Binary cross-entropy loss for binary classification
    metrics=['accuracy']                   # Track accuracy during training
)
