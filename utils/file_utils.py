"""
File utilities for AI Document Processing Service
"""

import os
import mimetypes

def detect_file_type(file_content: bytes, filename: str) -> str:
    """Detect file type from content and filename"""
    # First try to detect from filename extension
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    # Map extensions to MIME types
    extension_to_mime = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xls': 'application/vnd.ms-excel',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    }
    
    # Check if extension is supported
    if file_extension in extension_to_mime:
        return file_extension
    
    # If extension not found, try to detect from content (basic magic bytes)
    if file_content.startswith(b'%PDF'):
        return 'pdf'
    elif file_content.startswith(b'PK\x03\x04'):
        # ZIP-based format (DOCX, XLSX, PPTX)
        # This is a simplified detection - in real scenario you'd check internal structure
        return file_extension if file_extension in ['docx', 'xlsx', 'pptx'] else 'unknown'
    else:
        return 'unknown'

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return os.path.splitext(filename)[1].lower()

def is_supported_file_type(filename: str) -> bool:
    """Check if file type is supported"""
    from config.settings import SUPPORTED_FILE_EXTENSIONS
    extension = get_file_extension(filename).lstrip('.')
    return extension in SUPPORTED_FILE_EXTENSIONS

def validate_file_size(file_size: int) -> bool:
    """Validate file size against limits"""
    from config.settings import MAX_FILE_SIZE
    return file_size <= MAX_FILE_SIZE
