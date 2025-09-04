"""
AI Models management for Document Processing Service
"""

import torch
from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# Try to import specialized models
try:
    from nudenet import NudeDetector
    NUDENET_AVAILABLE = True
    print("NudeNet library detected - High accuracy NSFW detection available")
except ImportError:
    NUDENET_AVAILABLE = False
    print("NudeNet not available. Install with: pip install nudenet")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    EDUCATIONAL_MODELS_AVAILABLE = True
    print("Educational moderation models available")
except ImportError:
    EDUCATIONAL_MODELS_AVAILABLE = False
    print("Educational models not available")

from config.settings import MODEL_NAMES

class AIModelManager:
    """Manages all AI models for the service"""
    
    def __init__(self):
        self.nudenet_detector = None
        self.clip_model = None
        self.clip_processor = None
        self.falconsai_nsfw_classifier = None
        self.toxicity_pipeline = None  # For English text moderation
        
    def load_all_models(self):
        """Load all available AI models"""
        print("Initializing AI Service...")
        
        # Load NudeNet for NSFW detection
        self._load_nudenet()
        
        # Load Falconsai NSFW classifier
        self._load_falconsai()
        
        # Load CLIP for general image classification
        self._load_clip()
        
        # Load English text moderation model
        self._load_english_toxicity_model()
        
        print("AI Service initialization completed")
    
    def _load_nudenet(self):
        """Load NudeNet detector"""
        if NUDENET_AVAILABLE:
            try:
                print("Loading NudeNet detector...")
                self.nudenet_detector = NudeDetector()
                print("NudeNet detector loaded successfully")
            except Exception as e:
                print(f"Warning: NudeNet failed to load: {e}")
                self.nudenet_detector = None
    
    def _load_falconsai(self):
        """Load Falconsai NSFW classifier"""
        try:
            print("Loading Falconsai NSFW classifier...")
            self.falconsai_nsfw_classifier = pipeline(
                "image-classification", 
                model=MODEL_NAMES["nsfw"]
            )
            print("Falconsai NSFW classifier loaded successfully")
        except Exception as e:
            print(f"Warning: Falconsai NSFW classifier failed to load: {e}")
            self.falconsai_nsfw_classifier = None
    
    def _load_clip(self):
        """Load CLIP model"""
        try:
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained(MODEL_NAMES["clip"])
            self.clip_processor = CLIPProcessor.from_pretrained(MODEL_NAMES["clip"])
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Warning: CLIP failed to load: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def _load_english_toxicity_model(self):
        """Load English toxicity model for text moderation"""
        try:
            print("Loading English toxicity model...")
            self.toxicity_pipeline = pipeline(
                "text-classification", 
                model=MODEL_NAMES["toxicity"],
                device=0 if torch.cuda.is_available() else -1
            )
            print("English toxicity model loaded successfully")
        except Exception as e:
            print(f"Warning: English toxicity model failed to load: {e}")
            self.toxicity_pipeline = None
    
    def get_models_status(self):
        """Get status of all loaded models"""
        return {
            "falconsai_nsfw_classifier": self.falconsai_nsfw_classifier is not None,
            "nudenet_detector": self.nudenet_detector is not None,
            "clip_model": self.clip_model is not None,
            "toxicity_pipeline": self.toxicity_pipeline is not None,
            "vietnamese_moderation": "keyword_based"  # Always available
        }
    
    def get_capabilities(self):
        """Get service capabilities based on loaded models"""
        return {
            "nsfw_detection": "Falconsai" if self.falconsai_nsfw_classifier else ("NudeNet" if self.nudenet_detector else "Basic"),
            "violence_detection": "CLIP" if self.clip_model else "Basic",
            "text_moderation": "Hybrid (AI + Keywords)" if self.toxicity_pipeline else "Keywords Only",
            "educational_content_moderation": "Hybrid (AI + Keywords)" if self.toxicity_pipeline else "Keywords Only"
        }
    
    def get_accuracy_estimates(self):
        """Get accuracy estimates for loaded models"""
        return {
            "nsfw_detection": "92-97%" if self.falconsai_nsfw_classifier else ("90-95%" if self.nudenet_detector else "60-70%"),
            "violence_detection": "75-85%" if self.clip_model else "50-60%",
            "text_moderation": "85-90% (Hybrid)" if self.toxicity_pipeline else "70-80% (Keywords)",
            "educational_moderation": "85-90% (Hybrid)" if self.toxicity_pipeline else "70-80% (Keywords)"
        }

# Global model manager instance
model_manager = AIModelManager()
