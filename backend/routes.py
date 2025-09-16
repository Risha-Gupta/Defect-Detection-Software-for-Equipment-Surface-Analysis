# Import FastAPI components for creating API routes and handling requests
from fastapi import APIRouter, File, UploadFile, HTTPException, Form

# Import TensorFlow for loading and using the pre-trained models
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

# Global dictionary to store loaded models
models = {}

script_dir = os.path.dirname(os.path.abspath(__file__))
# Function to load a specific model
def load_model(model_name):
    global models
    
    if model_name not in models:
        # Define model paths
        if model_name == "simple_cnn":
            model_path = os.path.join(script_dir,"..", "ml-model", "car_damage_classifier.keras")
        elif model_name == "complex_cnn":
            model_path = os.path.join(script_dir,"..", "ml-model", "complex_cnn_canny.keras")
        elif model_name == "mobilenet_v2":
            model_path = os.path.join(script_dir,"..", "ml-model", "mobilenetv2_damage_classifier.keras")
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        try:
            # Load the model
            models[model_name] = tf.keras.models.load_model(model_path, compile=False)
            print(f"Model '{model_name}' loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model '{model_name}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return models[model_name]

def preprocess_image(image_bytes, model_name):
    try:
        # Open the image from bytes using PIL
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to RGB format
        img = img.convert('RGB')
        
        img = img.resize((150, 150))
        # Convert to array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# API endpoint for prediction with model selection
@router.post("/predict")
async def predict_damage(file: UploadFile = File(...), model: str = Form("simple_cnn")):
    
    # Check if file was uploaded
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check if file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read the uploaded file
        image_bytes = await file.read()
        
        # Load the selected model
        current_model = load_model(model)
        print("hi")
        # Preprocess the image
        processed_image = preprocess_image(image_bytes, model)
        print(f"Processed image shape: {processed_image.shape}")

        prediction = current_model.predict(processed_image)
        print(f"Raw model output: {prediction}")
        prediction_prob = float(prediction[0][0])
        
        if prediction_prob > 0.5:
            label = "damaged"
            confidence = prediction_prob
        else:
            label = "not damaged"
            confidence = 1 - prediction_prob
        
        return {
            "label": label,
            "confidence": confidence,
            "model_used": model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
