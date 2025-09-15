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

# Function to preprocess uploaded image for model prediction
def preprocess_image(image_bytes):
    try:
        # Open the image from bytes using PIL
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to RGB format (removes alpha channel if present)
        img = img.convert('RGB')
        
        # Resize image to 150x150 pixels as required by the model
        img = img.resize((150, 150))
        
        # Convert PIL image to NumPy array
        img_array = np.array(img)
        
        # Normalize pixel values from 0-255 range to 0-1 range
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension (model expects shape: (batch_size, height, width, channels))
        img_array = np.expand_dims(img_array, axis=0)
        
        # Return the preprocessed image array
        return img_array
        
    except Exception as e:  # Handle any errors during image preprocessing
        # Raise HTTP exception for image processing errors
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# API endpoint for car damage prediction
@router.post("/predict")
async def predict_damage(file: UploadFile = File(...)):
    # Check if a file was actually uploaded
    if not file:
        # Return error if no file provided
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check if uploaded file is an image by examining content type
    if not file.content_type.startswith('image/'):
        # Return error if file is not an image
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read the uploaded file content as bytes
        image_bytes = await file.read()
        
        # Load the pre-trained model (loads once, then reuses)
        current_model = load_model()
        
        # Preprocess the uploaded image for model input
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction using the loaded model
        prediction = current_model.predict(processed_image)
        
        # Extract prediction probability (sigmoid output between 0 and 1)
        prediction_prob = float(prediction[0][0])
        
        # Convert probability to binary classification (threshold at 0.5)
        if prediction_prob > 0.5:
            # High probability indicates damage
            result = "Damaged"
            # Calculate confidence as percentage above 50%
            confidence = prediction_prob * 100
        else:
            # Low probability indicates no damage
            result = "Not Damaged"
            # Calculate confidence as percentage below 50% (inverted)
            confidence = (1 - prediction_prob) * 100
        
        # Return JSON response with prediction results
        return {
            "prediction": result,  # Classification result
            "confidence": round(confidence, 2),  # Confidence percentage rounded to 2 decimals
            "raw_probability": round(prediction_prob, 4),  # Raw model output
            "filename": file.filename  # Original filename for reference
        }
        
    except Exception as e:  # Handle any unexpected errors during prediction
        # Return server error with details
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
