import os
from typing import Dict, Any

class Config:
    """Configuration class for AI Service"""
    
    # Service Configuration
    SERVICE_NAME = "AI Document Processing Service"
    SERVICE_VERSION = "2.0.0"
    HOST = os.getenv("AI_SERVICE_HOST", "0.0.0.0")
    PORT = int(os.getenv("AI_SERVICE_PORT", "8000"))
    
    # Model Configuration
    MODELS = {
        "toxicity": {
            "name": "unitary/toxic-bert",
            "enabled": os.getenv("ENABLE_TOXICITY_MODEL", "true").lower() == "true"
        },
        "zero_shot": {
            "name": "facebook/bart-large-mnli",
            "enabled": os.getenv("ENABLE_ZERO_SHOT_MODEL", "true").lower() == "true"
        },
        "nsfw": {
            "name": "Falconsai/nsfw_image_detection",
            "enabled": os.getenv("ENABLE_NSFW_MODEL", "true").lower() == "true"
        },
        "clip": {
            "name": "openai/clip-vit-base-patch32",
            "enabled": os.getenv("ENABLE_CLIP_MODEL", "true").lower() == "true"
        },
        "embedding": {
            "name": "all-MiniLM-L6-v2",
            "enabled": os.getenv("ENABLE_EMBEDDING_MODEL", "true").lower() == "true"
        }
    }
    
    # Thresholds
    THRESHOLDS = {
        "nsfw": float(os.getenv("NSFW_THRESHOLD", "0.75")),
        "violence_image": float(os.getenv("VIOLENCE_IMG_THRESHOLD", "0.60")),
        "toxicity": float(os.getenv("TOXICITY_THRESHOLD", "0.70")),
        "sexual_text": float(os.getenv("SEXUAL_TEXT_THRESHOLD", "0.65")),
        "violence_text": float(os.getenv("VIOLENCE_TEXT_THRESHOLD", "0.60"))
    }
    
    # Processing Configuration
    PROCESSING = {
        "max_text_length": int(os.getenv("MAX_TEXT_LENGTH", "1000")),
        "max_embedding_length": int(os.getenv("MAX_EMBEDDING_LENGTH", "512")),
        "chunk_size_mb": int(os.getenv("CHUNK_SIZE_MB", "5")),
        "num_permutations": int(os.getenv("MINHASH_PERMUTATIONS", "128")),
        "simhash_distance": int(os.getenv("SIMHASH_DISTANCE", "10"))
    }
    
    # Performance Configuration
    PERFORMANCE = {
        "enable_gpu": os.getenv("ENABLE_GPU", "false").lower() == "true",
        "max_workers": int(os.getenv("MAX_WORKERS", "4")),
        "batch_size": int(os.getenv("BATCH_SIZE", "10")),
        "timeout_seconds": int(os.getenv("TIMEOUT_SECONDS", "300"))
    }
    
    # Logging Configuration
    LOGGING = {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": os.getenv("LOG_FILE", "ai_service.log")
    }
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        return cls.MODELS.get(model_type, {})
    
    @classmethod
    def is_model_enabled(cls, model_type: str) -> bool:
        """Check if specific model is enabled"""
        model_config = cls.get_model_config(model_type)
        return model_config.get("enabled", False)
    
    @classmethod
    def get_threshold(cls, threshold_type: str) -> float:
        """Get threshold value for specific type"""
        return cls.THRESHOLDS.get(threshold_type, 0.5)
    
    @classmethod
    def get_processing_config(cls, config_type: str) -> Any:
        """Get processing configuration value"""
        return cls.PROCESSING.get(config_type, None)
    
    @classmethod
    def get_performance_config(cls, config_type: str) -> Any:
        """Get performance configuration value"""
        return cls.PERFORMANCE.get(config_type, None)

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    # Disable heavy models for testing
    MODELS = {
        "toxicity": {"name": "unitary/toxic-bert", "enabled": False},
        "zero_shot": {"name": "facebook/bart-large-mnli", "enabled": False},
        "nsfw": {"name": "Falconsai/nsfw_image_detection", "enabled": False},
        "clip": {"name": "openai/clip-vit-base-patch32", "enabled": False},
        "embedding": {"name": "all-MiniLM-L6-v2", "enabled": False}
    }

# Configuration factory
def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Global config instance
config = get_config()


