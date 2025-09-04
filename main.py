"""
Main FastAPI application for AI Document Processing Service
"""

import time
import base64
from typing import Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO

# Import our modules
from models.ai_models import model_manager
from models.data_models import (
    ChunkProcessingRequest, ChunkProcessingResult,
    FileProcessingRequest, FileProcessingResult
)
from services.text_moderation import moderate_text_advanced, moderate_educational_content
from services.image_moderation import moderate_image_advanced
from services.file_processing import extract_file_content, moderate_file_content
from utils.text_utils import compute_text_hash, compute_simhash, compute_minhash_signature
from config.settings import (
    SERVICE_NAME, SERVICE_VERSION, HOST, PORT,
    FILE_PROCESSING_THRESHOLDS
)

# Initialize FastAPI app
app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="AI-powered document moderation service for educational content"
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
    accuracy_estimates = model_manager.get_accuracy_estimates()
    
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "models_loaded": models_status,
        "capabilities": capabilities,
        "accuracy_estimates": accuracy_estimates,
        "features": {
            "text_moderation": True,
            "image_moderation": True,
            "educational_content_moderation": True,
            "file_processing": True,
            "vietnamese_support": True,
            "chunk_processing": True
        }
    }

# Text moderation endpoints
@app.post("/moderate-text")
async def moderate_text(text: str):
    """Moderate a single text input"""
    try:
        result = moderate_text_advanced(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/moderate-educational-text")
async def moderate_educational_text(text: str):
    """Moderate educational content text"""
    try:
        result = moderate_educational_content(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Image moderation endpoints
@app.post("/moderate-image")
async def moderate_image(file: UploadFile = File(...)):
    """Moderate a single image file"""
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        result = moderate_image_advanced(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/moderate-image-base64")
async def moderate_image_base64(image_data: str):
    """Moderate a single image (base64 encoded)"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        result = moderate_image_advanced(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chunk processing endpoints
@app.post("/process-chunks", response_model=ChunkProcessingResult)
async def process_chunks(request: ChunkProcessingRequest):
    """Process multiple text and image chunks"""
    start_time = time.time()
    
    try:
        text_results = []
        image_results = []
        all_scores = []
        
        # Process text chunks
        for chunk_data in request.text_chunks:
            chunk_index = chunk_data.get("chunk_index", 0)
            text_content = chunk_data.get("text_content", "")
            
            if text_content.strip():
                result = moderate_educational_content(text_content)
                result["chunk_index"] = chunk_index
                text_results.append(result)
                
                # Collect scores
                all_scores.extend([
                    result.get("inappropriate_score", 0.0),
                    result.get("offensive_score", 0.0),
                    result.get("sexual_content_score", 0.0)
                ])
        
        # Process image chunks
        for chunk_data in request.image_chunks:
            chunk_index = chunk_data.get("chunk_index", 0)
            image_data = chunk_data.get("image_data", "")
            
            if image_data:
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes))
                    
                    result = moderate_image_advanced(image)
                    result["chunk_index"] = chunk_index
                    image_results.append(result)
                    
                    # Collect scores
                    all_scores.extend([
                        result.get("nsfw_score", 0.0),
                        result.get("violence_score", 0.0)
                    ])
                    
                except Exception as e:
                    print(f"Image chunk {chunk_index} processing failed: {e}")
                    image_results.append({
                        "chunk_index": chunk_index,
                        "error": str(e),
                        "nsfw_score": 0.0,
                        "violence_score": 0.0
                    })
        
        # Calculate overall score
        overall_score = max(all_scores) if all_scores else 0.0
        
        processing_time = time.time() - start_time
        
        return ChunkProcessingResult(
            file_id=request.file_id,
            processing_status="completed",
            overall_moderation_score=overall_score,
            processing_time=processing_time,
            text_results=text_results,
            image_results=image_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# File processing endpoints
@app.post("/process-file", response_model=FileProcessingResult)
async def process_file(
    file: UploadFile = File(...),
    thresholds: str = Form(None),
    processing_options: str = Form(None)
):
    """Process uploaded file (PDF, DOCX, XLSX, PPTX)"""
    start_time = time.time()
    
    try:
        # Parse thresholds if provided
        parsed_thresholds = FILE_PROCESSING_THRESHOLDS.copy()
        if thresholds:
            import json
            try:
                parsed_thresholds.update(json.loads(thresholds))
            except json.JSONDecodeError:
                pass
        
        # Parse processing options if provided
        parsed_options = {}
        if processing_options:
            import json
            try:
                parsed_options = json.loads(processing_options)
            except json.JSONDecodeError:
                pass
        
        # Read file content
        file_content = await file.read()
        
        # Extract content from file
        extraction_result = extract_file_content(file_content, file.filename)
        
        if extraction_result["extraction_status"] != "success":
            raise HTTPException(
                status_code=400, 
                detail=f"File extraction failed: {extraction_result.get('error', 'Unknown error')}"
            )
        
        # Moderate extracted content
        moderation_result = moderate_file_content(
            extraction_result["text_content"],
            extraction_result["images"],
            parsed_thresholds
        )
        
        processing_time = time.time() - start_time
        
        return FileProcessingResult(
            file_id=parsed_options.get("file_id", "unknown"),
            filename=file.filename,
            file_type=extraction_result["file_type"],
            processing_status="completed",
            overall_moderation_score=moderation_result["overall_score"],
            processing_time=processing_time,
            text_content=extraction_result["text_content"][:1000] + "..." if len(extraction_result["text_content"]) > 1000 else extraction_result["text_content"],
            text_moderation_result=moderation_result["text_moderation"],
            image_count=len(extraction_result["images"]),
            image_moderation_results=moderation_result["image_moderation"],
            thresholds_used=parsed_thresholds,
            details={
                "extraction_status": extraction_result["extraction_status"],
                "text_length": len(extraction_result["text_content"]),
                "overall_risk": moderation_result["overall_risk"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-file-with-request", response_model=FileProcessingResult)
async def process_file_with_request(
    file: UploadFile = File(...),
    request_data: str = Form(...)
):
    """Process file with detailed request data"""
    try:
        import json
        request_info = json.loads(request_data)
        
        # Create a temporary file processing request
        file_processing_request = FileProcessingRequest(**request_info)
        
        # Process the file using the existing endpoint logic
        return await process_file(
            file=file,
            thresholds=json.dumps(file_processing_request.thresholds) if file_processing_request.thresholds else None,
            processing_options=json.dumps(file_processing_request.processing_options) if file_processing_request.processing_options else None
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request_data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.post("/compute-hashes")
async def compute_hashes(text: str):
    """Compute various hashes for text"""
    try:
        return {
            "text_hash": compute_text_hash(text),
            "simhash": compute_simhash(text),
            "minhash_signature": compute_minhash_signature(text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
