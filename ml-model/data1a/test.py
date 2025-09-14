# Import TensorFlow library for deep learning functionality
import tensorflow as tf

# Import Keras modules for building neural networks
from tensorflow import keras

# Import Sequential model class for building linear stack of layers
from tensorflow.keras.models import Sequential

# Import layer types needed for CNN architecture
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

pic_height = 150  # Height in pixels
pic_width = 150   # Width in pixels
color_channels = 3  # Number of color channels (RGB)

batch_count = 32

training_rounds = 30

num_classes = 2

train_data_prep = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values from 0-255 to 0-1 range
    rotation_range=20,       # Randomly rotate images up to 20 degrees
    width_shift_range=0.2,   # Randomly shift images horizontally by 20%
    height_shift_range=0.2,  # Randomly shift images vertically by 20%
    shear_range=0.2,         # Apply random shear transformations
    zoom_range=0.2,          # Randomly zoom into images by up to 20%
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest'      # Fill pixels after transformations using nearest neighbor
)

val_data_prep = ImageDataGenerator(
    rescale=1./255           # Only normalize pixel values, no augmentation for validation
)

train_pics = train_data_prep.flow_from_directory(
    'training',                    # Directory containing training subdirectories
    target_size=(pic_height, pic_width),  # Resize all images to specified dimensions
    batch_size=batch_count,         # Number of images to load per batch
    class_mode='binary',           # Binary classification (0 or 1)
    shuffle=True                   # Randomly shuffle the training data
)

val_pics = val_data_prep.flow_from_directory(
    'validation',                  # Directory containing validation subdirectories
    target_size=(pic_height, pic_width),  # Resize all images to specified dimensions
    batch_size=batch_count,         # Number of images to load per batch
    class_mode='binary',           # Binary classification (0 or 1)
    shuffle=False                  # Don't shuffle validation data for consistent evaluation
)

# Print information about the data generators
print(f"Training samples: {train_pics.samples}")      # Total number of training images
print(f"Validation samples: {val_pics.samples}") # Total number of validation images
print(f"Class indices: {train_pics.class_indices}")   # Mapping of class names to indices

my_model = Sequential()

# First convolutional block
my_model.add(Conv2D(
    32,                           # Number of filters (feature detectors)
    (3, 3),                      # Filter size (3x3 kernel)
    activation='relu',           # ReLU activation function
    input_shape=(pic_height, pic_width, color_channels)  # Input image dimensions
))
my_model.add(MaxPooling2D(2, 2))    # Max pooling with 2x2 pool size to reduce dimensions

# Second convolutional block
my_model.add(Conv2D(
    64,                          # Increase number of filters to 64
    (3, 3),                      # 3x3 kernel size
    activation='relu'            # ReLU activation function
))
my_model.add(MaxPooling2D(2, 2))    # Max pooling to further reduce dimensions

# Third convolutional block
my_model.add(Conv2D(
    128,                         # Increase number of filters to 128
    (3, 3),                      # 3x3 kernel size
    activation='relu'            # ReLU activation function
))
my_model.add(MaxPooling2D(2, 2))    # Max pooling layer

# Fourth convolutional block
my_model.add(Conv2D(
    128,                         # Keep 128 filters
    (3, 3),                      # 3x3 kernel size
    activation='relu'            # ReLU activation function
))
my_model.add(MaxPooling2D(2, 2))    # Final max pooling layer

# Flatten the 2D feature maps into 1D vector for dense layers
my_model.add(Flatten())

# Add dropout layer to prevent overfitting
my_model.add(Dropout(0.5))          # Randomly set 50% of inputs to 0 during training

# First fully connected (dense) layer
my_model.add(Dense(
    512,                         # 512 neurons in this layer
    activation='relu'            # ReLU activation function
))

# Final output layer for binary classification
my_model.add(Dense(
    1,                           # Single neuron for binary output
    activation='sigmoid'         # Sigmoid activation for probability output (0-1)
))

# Compile the model with optimizer, loss function, and metrics
my_model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adam optimizer with learning rate 0.001
    loss='binary_crossentropy',           # Binary cross-entropy loss for binary classification
    metrics=['accuracy']                  # Track accuracy during training
)

# Print model architecture summary
print("\nModel Architecture:")
my_model.summary()                          # Display detailed model structure

train_batches = train_pics.samples // batch_count      # Number of batches per training epoch
val_batches = val_pics.samples // batch_count  # Number of batches per validation epoch

# Print training information
print(f"\nTraining steps per epoch: {train_batches}")
print(f"Validation steps per epoch: {val_batches}")
print(f"Starting training for {training_rounds} epochs...")

training_log = my_model.fit(
    train_pics,             # Training data generator
    steps_per_epoch=train_batches, # Number of steps per epoch
    epochs=training_rounds,  # Use the training_rounds variable (30 epochs)
    validation_data=val_pics,  # Validation data generator
    validation_steps=val_batches,     # Number of validation steps per epoch
    verbose=1                    # Print progress during training
)

# Print training completion message
print(f"\nTraining completed after {training_rounds} epochs!")

saved_model_name = 'car_damage_classifier.h5'  # Define filename for saved model
my_model.save(saved_model_name)                    # Save the entire model (architecture + weights)

# Print confirmation of model saving
print(f"Model saved as '{saved_model_name}' in the current directory.")

final_train_acc = training_log.history['accuracy'][-1]      # Get last training accuracy
final_val_acc = training_log.history['val_accuracy'][-1]    # Get last validation accuracy

print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# Print model file information
if os.path.exists(saved_model_name):          # Check if model file was created successfully
    file_size = os.path.getsize(saved_model_name) / (1024 * 1024)  # Get file size in MB
    print(f"Model file size: {file_size:.2f} MB")
    print("Model is ready for use!")
else:
    print("Error: Model file was not created successfully.")
