"""
Text utilities for AI Document Processing Service
"""

import hashlib
import numpy as np
from datasketch import MinHash
from simhash import Simhash
from config.settings import NUM_PERM

def safe_convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [safe_convert_to_python_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_convert_to_python_types(value) for key, value in obj.items()}
    else:
        return obj

def compute_text_hash(text: str) -> str:
    """Compute SHA256 hash of text"""
    return hashlib.sha256(text.encode()).hexdigest()

def compute_simhash(text: str) -> str:
    """Compute SimHash of text"""
    try:
        simhash = Simhash(text)
        return str(simhash.value)
    except Exception as e:
        print(f"SimHash computation error: {e}")
        return "0"

def compute_minhash_signature(text: str) -> list:
    """Compute MinHash signature with proper type conversion"""
    try:
        minhash = MinHash(num_perm=NUM_PERM)
        words = text.lower().split()
        for word in words:
            minhash.update(word.encode('utf8'))
        return [int(x) for x in minhash.hashvalues]
    except Exception as e:
        print(f"MinHash computation error: {e}")
        return [0] * NUM_PERM

def detect_language(text: str) -> str:
    """Simple language detection for Vietnamese vs English"""
    vietnamese_chars = ['ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư', 'à', 'á', 'ả', 'ã', 'ạ', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 
                       'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ì', 'í', 
                       'ỉ', 'ĩ', 'ị', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'ờ', 'ớ', 'ở', 'ỡ', 
                       'ợ', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ']
    
    vietnamese_count = sum(1 for char in text.lower() if char in vietnamese_chars)
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return "unknown"
    
    vietnamese_ratio = vietnamese_count / total_chars
    return "vietnamese" if vietnamese_ratio > 0.1 else "english"
