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
