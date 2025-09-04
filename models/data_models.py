"""
Pydantic data models for AI Document Processing Service
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TextChunk:
    chunk_index: int
    text_content: str
    chunk_size: int
    metadata: Dict[str, Any]

@dataclass  
class ImageChunk:
    chunk_index: int
    image_data: str  # base64
    image_format: str
    image_size: List[int]
    metadata: Dict[str, Any]

class ChunkProcessingRequest(BaseModel):
    file_id: str
    file_type: str
    original_filename: str
    text_chunks: List[Dict[str, Any]] = []
    image_chunks: List[Dict[str, Any]] = []
    processing_options: Dict[str, Any] = {}

class ChunkProcessingResult(BaseModel):
    file_id: str
    processing_status: str
    overall_moderation_score: float
    processing_time: float
    text_results: List[Dict[str, Any]] = []
    image_results: List[Dict[str, Any]] = []

class FileProcessingRequest(BaseModel):
    file_id: str
    filename: str
    file_type: str
    thresholds: Dict[str, float] = {}
    processing_options: Dict[str, Any] = {}

class ViolationInfo(BaseModel):
    page_number: int
    violation_type: str  # "text", "image", "both"
    severity: str  # "HIGH", "MEDIUM", "LOW"
    score: float
    details: str = ""

class FileProcessingResult(BaseModel):
    file_id: str
    filename: str
    file_type: str
    processing_status: str
    overall_moderation_score: float
    processing_time: float
    
    # Summary information
    total_pages: int = 0
    violations_found: int = 0
    violation_pages: List[int] = []
    violation_summary: List[ViolationInfo] = []
    
    # Detailed results (can be limited)
    text_content: str = ""
    text_moderation_result: Dict[str, Any] = {}
    image_count: int = 0
    image_moderation_results: List[Dict[str, Any]] = []
    thresholds_used: Dict[str, float] = {}
    details: Dict[str, Any] = {}
    
    # Response optimization flags
    include_full_text: bool = False
    include_detailed_results: bool = False
