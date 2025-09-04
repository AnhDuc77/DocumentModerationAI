"""
Main FastAPI application for AI Document Processing Service
Simplified API with minimal input/output
"""

import time
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO

# Import our modules
from models.ai_models import model_manager
from services.text_moderation import moderate_text_hybrid
from services.image_moderation import moderate_image_advanced
from services.file_processing import extract_file_content, moderate_file_content
from config.settings import SERVICE_NAME, SERVICE_VERSION, HOST, PORT

# Initialize FastAPI app
app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="AI-powered document moderation service - simplified API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI models on startup
@app.on_event("startup")
async def startup_event():
    """Load AI models when the service starts"""
    model_manager.load_all_models()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with service status"""
    models_status = model_manager.get_models_status()
    capabilities = model_manager.get_capabilities()
    
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "models_loaded": models_status,
        "capabilities": capabilities
    }

# Main file processing endpoint - simplified
@app.post("/check-file")
async def check_file(file: UploadFile = File(...)):
    """Check file for inappropriate content - simplified API"""
    start_time = time.time()
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract content from file
        extraction_result = extract_file_content(file_content, file.filename)
        
        if extraction_result["extraction_status"] != "success":
            return {
                "status": "error",
                "message": f"File extraction failed: {extraction_result.get('error', 'Unknown error')}"
            }
        
        # Moderate extracted content
        moderation_result = moderate_file_content(
            extraction_result["text_content"],
            extraction_result["images"]
        )
        
        processing_time = time.time() - start_time
        
        # Create simplified response
        score = moderation_result["overall_score"]
        if score < 0.3:
            status = "safe"
        elif score < 0.7:
            status = "warning"
        else:
            status = "danger"
        
        violations = moderation_result.get("violations_found", 0)
        pages = moderation_result.get("violation_pages", [])
        
        return {
            "status": status,
            "score": round(score, 2),
            "violations": violations,
            "pages": pages[:5] if pages else [],  # Limit to first 5 pages
            "processing_time": round(processing_time, 2)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Text moderation endpoint - simplified
@app.post("/check-text")
async def check_text(text: str):
    """Check text for inappropriate content - simplified API"""
    try:
        result = moderate_text_hybrid(text)
        
        score = max(
            result.get("inappropriate_score", 0.0),
            result.get("offensive_score", 0.0),
            result.get("sexual_content_score", 0.0),
            result.get("violence_score", 0.0)
        )
        
        if score < 0.3:
            status = "safe"
        elif score < 0.7:
            status = "warning"
        else:
            status = "danger"
        
        return {
            "status": status,
            "score": round(score, 2),
            "language": result.get("language_detected", "unknown")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Image moderation endpoint - simplified
@app.post("/check-image")
async def check_image(file: UploadFile = File(...)):
    """Check image for inappropriate content - simplified API"""
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        result = moderate_image_advanced(image)
        
        score = max(
            result.get("nsfw_score", 0.0),
            result.get("violence_score", 0.0)
        )
        
        if score < 0.3:
            status = "safe"
        elif score < 0.7:
            status = "warning"
        else:
            status = "danger"
        
        return {
            "status": status,
            "score": round(score, 2),
            "nsfw_score": round(result.get("nsfw_score", 0.0), 2),
            "violence_score": round(result.get("violence_score", 0.0), 2)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
