# Import FastAPI class for creating the main application instance
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import the router containing our prediction endpoints
from routes import router
import os
# Create the main FastAPI application instance
app = FastAPI(
    title="Car Damage Classifier API",  # API title shown in documentation
    description="API for predicting car damage from uploaded images",  # API description
    version="1.0.0"  # API version number
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routes from routes.py in the main application
app.include_router(router)



# Root endpoint that returns a welcome message
@app.get("/")
async def root():
    # Return a simple JSON response with welcome message
    return {
        "message": "Welcome to the Car Damage Classifier API",  # Welcome message
        "version": "1.0.0",  # API version
        "endpoints": {  # Available endpoints information
            "/": "This welcome message",  # Root endpoint description
            "/predict": "POST endpoint for car damage prediction",  # Prediction endpoint description
            "/docs": "Interactive API documentation"  # Swagger UI documentation
        }
    }

# Health check endpoint to verify API is running
@app.get("/health")
async def health_check():
    # Return simple health status
    return {"status": "healthy", "message": "API is running successfully"}

# Run the application if this file is executed directly
if __name__ == "__main__":
    # Import uvicorn server for running the FastAPI application
    import uvicorn
    
    # Start the server with specified configuration
    uvicorn.run(
        "main:app",  # Application module and instance
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,  # Port number for the API
        reload=True  # Enable auto-reload during development
    )
