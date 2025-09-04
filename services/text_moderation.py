"""
AI-only Text moderation services for AI Document Processing Service
Uses only martin-ha/toxic-comment-model for all languages
"""

from typing import Dict, Any
from utils.text_utils import safe_convert_to_python_types, detect_language
from models.ai_models import model_manager

def moderate_text_ai_only(text: str) -> Dict[str, Any]:
    """AI-only text moderation using martin-ha/toxic-comment-model for all languages"""
    results = {
        "inappropriate_score": 0.0,
        "offensive_score": 0.0,
        "sexual_content_score": 0.0,
        "violence_score": 0.0,
        "overall_risk": "LOW",
        "language_detected": detect_language(text),
        "moderation_method": "ai_model",
        "details": {}
    }

    try:
        if model_manager.toxicity_pipeline:
            # Use AI model for all languages
            toxicity_result = model_manager.toxicity_pipeline(text)

            if isinstance(toxicity_result, list) and len(toxicity_result) > 0:
                for result in toxicity_result:
                    if result.get('label') in ['TOXIC', 'toxic', '1', 1]:
                        toxic_score = float(result.get('score', 0.0))
                        results["inappropriate_score"] = toxic_score
                        results["offensive_score"] = toxic_score
                        # Map toxicity to other categories
                        results["sexual_content_score"] = toxic_score * 0.8
                        results["violence_score"] = toxic_score * 0.6
                        results["moderation_method"] = "ai_model"
                        break
                    elif result.get('label') in ['CLEAN', 'clean', '0', 0]:
                        toxic_score = 1.0 - float(result.get('score', 0.0))
                        results["inappropriate_score"] = toxic_score
                        results["offensive_score"] = toxic_score
                        results["sexual_content_score"] = toxic_score * 0.8
                        results["violence_score"] = toxic_score * 0.6
                        results["moderation_method"] = "ai_model"
                        break

                results["details"]["ai_model_result"] = toxicity_result
            else:
                # No clear result, assume clean
                results["inappropriate_score"] = 0.0
                results["offensive_score"] = 0.0
                results["sexual_content_score"] = 0.0
                results["violence_score"] = 0.0
        else:
            # AI model not available, return clean
            print("Warning: AI model not available, returning clean result")
            results["inappropriate_score"] = 0.0
            results["offensive_score"] = 0.0
            results["sexual_content_score"] = 0.0
            results["violence_score"] = 0.0
            results["moderation_method"] = "ai_unavailable"

        # Overall risk assessment
        max_score = max(
            results["inappropriate_score"],
            results["offensive_score"],
            results["sexual_content_score"],
            results["violence_score"]
        )

        if max_score > 0.8:
            results["overall_risk"] = "HIGH"
        elif max_score > 0.5:
            results["overall_risk"] = "MEDIUM"
        else:
            results["overall_risk"] = "LOW"

        results = safe_convert_to_python_types(results)
        return results

    except Exception as e:
        print(f"AI text moderation failed: {e}")
        return {
            "inappropriate_score": 0.0,
            "offensive_score": 0.0,
            "sexual_content_score": 0.0,
            "violence_score": 0.0,
            "overall_risk": "ERROR",
            "language_detected": detect_language(text),
            "error": str(e)
        }

# Alias for backward compatibility
def moderate_text_simple(text: str) -> Dict[str, Any]:
    """Alias for moderate_text_ai_only for backward compatibility"""
    return moderate_text_ai_only(text)

def moderate_educational_content(text: str) -> Dict[str, Any]:
    """Alias for moderate_text_ai_only for backward compatibility"""
    return moderate_text_ai_only(text)

def moderate_text_advanced(text: str) -> Dict[str, Any]:
    """Alias for moderate_text_ai_only for backward compatibility"""
    return moderate_text_ai_only(text)

# Main function for new API
def moderate_text_hybrid(text: str) -> Dict[str, Any]:
    """Main function for new API - AI only"""
    return moderate_text_ai_only(text)