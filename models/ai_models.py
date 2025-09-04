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
        self.toxicity_pipeline = None
        self.hate_speech_pipeline = None
        self.falconsai_nsfw_classifier = None
        self.educational_offensive_classifier = None
        self.educational_inappropriate_classifier = None
        self.vietnamese_phobert_classifier = None
        
    def load_all_models(self):
        """Load all available AI models"""
        print("Initializing AI Service...")
        
        # Load NudeNet for NSFW detection
        self._load_nudenet()
        
        # Load Falconsai NSFW classifier
        self._load_falconsai()
        
        # Load CLIP for general image classification
        self._load_clip()
        
        # Load text moderation models
        self._load_text_models()
        
        # Load educational content moderation models
        self._load_educational_models()
        
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
    
    def _load_text_models(self):
        """Load text moderation models"""
        try:
            print("Loading text moderation models...")
            
            self.toxicity_pipeline = pipeline(
                "text-classification", 
                model=MODEL_NAMES["toxicity"],
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.hate_speech_pipeline = pipeline(
                "text-classification",
                model=MODEL_NAMES["hate_speech"],
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("Text moderation models loaded successfully")
            
        except Exception as e:
            print(f"Warning: Text models failed to load: {e}")
            self.toxicity_pipeline = None
            self.hate_speech_pipeline = None
    
    def _load_educational_models(self):
        """Load educational content moderation models"""
        if EDUCATIONAL_MODELS_AVAILABLE:
            try:
                print("Loading educational content moderation models...")
                
                # Load offensive content classifier
                self.educational_offensive_classifier = pipeline(
                    "text-classification",
                    model=MODEL_NAMES["educational_offensive"],
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Load inappropriate content classifier
                self.educational_inappropriate_classifier = pipeline(
                    "text-classification",
                    model=MODEL_NAMES["hate_speech"],  # Reuse hate speech model
                    device=0 if torch.cuda.is_available() else -1
                )
                
                print("Educational content moderation models loaded successfully")
                
            except Exception as e:
                print(f"Warning: Educational models failed to load: {e}")
                self.educational_offensive_classifier = None
                self.educational_inappropriate_classifier = None
    
    def get_models_status(self):
        """Get status of all loaded models"""
        return {
            "falconsai_nsfw_classifier": self.falconsai_nsfw_classifier is not None,
            "nudenet_detector": self.nudenet_detector is not None,
            "clip_model": self.clip_model is not None,
            "toxicity_pipeline": self.toxicity_pipeline is not None,
            "hate_speech_pipeline": self.hate_speech_pipeline is not None,
            "educational_offensive_classifier": self.educational_offensive_classifier is not None,
            "educational_inappropriate_classifier": self.educational_inappropriate_classifier is not None
        }
    
    def get_capabilities(self):
        """Get service capabilities based on loaded models"""
        return {
            "nsfw_detection": "Falconsai" if self.falconsai_nsfw_classifier else ("NudeNet" if self.nudenet_detector else "Basic"),
            "violence_detection": "CLIP" if self.clip_model else "Basic",
            "text_toxicity": "Advanced" if self.toxicity_pipeline else "Basic",
            "text_hate_speech": "Advanced" if self.hate_speech_pipeline else "Basic",
            "educational_content_moderation": "Advanced" if self.educational_offensive_classifier else "Keyword-based"
        }
    
    def get_accuracy_estimates(self):
        """Get accuracy estimates for loaded models"""
        return {
            "nsfw_detection": "92-97%" if self.falconsai_nsfw_classifier else ("90-95%" if self.nudenet_detector else "60-70%"),
            "violence_detection": "75-85%" if self.clip_model else "50-60%",
            "text_moderation": "85-90%" if self.toxicity_pipeline else "60-70%",
            "educational_moderation": "90-95%" if self.educational_offensive_classifier else "85-90%"
        }

# Global model manager instance
model_manager = AIModelManager()
