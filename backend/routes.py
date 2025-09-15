# Import FastAPI components for creating API routes and handling requests
from fastapi import APIRouter, File, UploadFile, HTTPException

# Import TensorFlow for loading and using the pre-trained model
import tensorflow as tf

# Import PIL (Python Imaging Library) for image processing
from PIL import Image

# Import NumPy for numerical operations on image arrays
import numpy as np

# Import io module for handling byte streams
import io

# Import os module for file path operations
import os

# Create an APIRouter instance to define route endpoints
router = APIRouter()

# Global variable to store the loaded model (will be loaded once)
model = None

# Function to load the pre-trained model from the specified path
def load_model():
    global model  # Access the global model variable
    if model is None:  # Check if model hasn't been loaded yet
        # Construct the path to the model file (going up one directory to ml-model folder)
        model_path = os.path.join("..", "ml-model", "car_damage_classifier.h5")
        try:
            # Load the pre-trained Keras model from the .h5 file
            model = tf.keras.models.load_model(model_path)
            # Print confirmation that model was loaded successfully
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:  # Handle any errors during model loading
            # Print error message if model loading fails
            print(f"Error loading model: {str(e)}")
            # Raise HTTP exception to inform client of server error
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    # Return the loaded model
    return model

