"""
Image moderation services for AI Document Processing Service
"""

from typing import Dict, Any
import torch
from PIL import Image
from models.ai_models import model_manager
from utils.text_utils import safe_convert_to_python_types

def moderate_image_with_falconsai(image: Image.Image) -> Dict[str, Any]:
    """Falconsai NSFW detection"""
    try:
        if not model_manager.falconsai_nsfw_classifier:
            return {"nsfw_score": 0.0, "detection_method": "falconsai_unavailable"}
        
        results = model_manager.falconsai_nsfw_classifier(image)
        
        nsfw_score = 0.0
        safe_score = 0.0
        
        for result in results:
            label = result['label'].lower()
            score = float(result['score'])
            
            if 'nsfw' in label or 'nude' in label or 'porn' in label or 'explicit' in label:
                nsfw_score = max(nsfw_score, score)
            elif 'safe' in label or 'normal' in label:
                safe_score = max(safe_score, score)
        
        final_nsfw_score = nsfw_score if nsfw_score > safe_score + 0.2 else 0.0
        
        return {
            "nsfw_score": float(final_nsfw_score),
            "detection_method": "Falconsai",
            "details": {
                "falconsai_raw": [{"label": r['label'], "score": float(r['score'])} for r in results],
                "nsfw_confidence": float(nsfw_score),
                "safe_confidence": float(safe_score)
            }
        }
        
    except Exception as e:
        print(f"Falconsai processing failed: {e}")
        return {"nsfw_score": 0.0, "detection_method": "falconsai_error", "error": str(e)}

def moderate_image_advanced(image: Image.Image) -> Dict[str, Any]:
    """Advanced image moderation with Falconsai + CLIP"""
    results = {
        "nsfw_score": 0.0,
        "violence_score": 0.0,
        "overall_risk": "LOW",
        "detection_method": "basic",
        "details": {}
    }
    
    try:
        # Falconsai for NSFW detection (preferred)
        if model_manager.falconsai_nsfw_classifier:
            try:
                falconsai_result = moderate_image_with_falconsai(image)
                results["nsfw_score"] = falconsai_result.get("nsfw_score", 0.0)
                results["details"].update(falconsai_result.get("details", {}))
                results["detection_method"] = "Falconsai"
            except Exception as e:
                print(f"Falconsai processing failed: {e}")
        
        # CLIP for violence detection
        if model_manager.clip_model and model_manager.clip_processor:
            try:
                violence_labels = [
                    "safe everyday image", 
                    "violent action", 
                    "dangerous weapon", 
                    "fighting or combat",
                    "graphic violence",
                    "war or conflict scene"
                ]
                
                inputs = model_manager.clip_processor(text=violence_labels, images=image, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model_manager.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                violence_scores = probs[0].tolist()
                
                safe_score = violence_scores[0]
                violent_scores = violence_scores[1:]
                max_violent_score = max(violent_scores)
                violence_confidence = max_violent_score - safe_score
                
                if violence_confidence > 0.3 and max_violent_score > 0.6:
                    violence_score = float(max_violent_score)
                else:
                    violence_score = 0.0
                
                results["violence_score"] = violence_score
                results["details"]["clip_violence_breakdown"] = dict(zip(violence_labels, violence_scores))
                results["details"]["violence_confidence"] = float(violence_confidence)
                results["details"]["safe_score"] = float(safe_score)
                
                if results["detection_method"] == "basic":
                    results["detection_method"] = "CLIP"
                else:
                    results["detection_method"] += " + CLIP"
                    
            except Exception as e:
                print(f"CLIP violence detection failed: {e}")
        
        # Overall assessment
        overall_score = max(results["nsfw_score"], results["violence_score"])
        
        if overall_score > 0.8:
            results["overall_risk"] = "HIGH"
        elif overall_score > 0.5:
            results["overall_risk"] = "MEDIUM"
        elif overall_score > 0.2:
            results["overall_risk"] = "LOW-MEDIUM"
        else:
            results["overall_risk"] = "LOW"
        
        results = safe_convert_to_python_types(results)
        return results
        
    except Exception as e:
        print(f"Image moderation processing failed: {e}")
        return {
            "nsfw_score": 0.0,
            "violence_score": 0.0,
            "overall_risk": "ERROR",
            "detection_method": "error",
            "error": str(e)
        }
