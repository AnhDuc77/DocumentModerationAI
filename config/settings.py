"""
Configuration settings for AI Document Processing Service
"""

# Model Configuration
NSFW_THRESHOLD = 0.7
VIOLENCE_THRESHOLD = 0.6
TOXICITY_THRESHOLD = 0.7
NUDENET_THRESHOLD = 0.6
NUM_PERM = 128

# Educational content moderation thresholds
EDUCATIONAL_INAPPROPRIATE_THRESHOLD = 0.6
EDUCATIONAL_OFFENSIVE_THRESHOLD = 0.7
EDUCATIONAL_SEXUAL_THRESHOLD = 0.8

# File processing thresholds (configurable)
FILE_PROCESSING_THRESHOLDS = {
    "inappropriate": 0.8,
    "offensive": 0.8,
    "sexual": 0.9,
    "violence": 0.8,
    "overall": 0.8
}

# Service Configuration
SERVICE_NAME = "AI Document Processing Service"
SERVICE_VERSION = "4.1.0"
HOST = "0.0.0.0"
PORT = 9000

# Supported file types
SUPPORTED_FILE_EXTENSIONS = ['pdf', 'docx', 'xlsx', 'xls', 'pptx']

# File size limits (in bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
CHUNK_SIZE = 1024 * 1024  # 1MB

# Model names
MODEL_NAMES = {
    "toxicity": "martin-ha/toxic-comment-model",
    "hate_speech": "unitary/toxic-bert",
    "nsfw": "Falconsai/nsfw_image_detection",
    "clip": "openai/clip-vit-base-patch32",
    "educational_offensive": "cardiffnlp/twitter-roberta-base-offensive"
}
