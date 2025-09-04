"""
Text moderation services for AI Document Processing Service
"""

from typing import Dict, Any
from models.ai_models import model_manager
from utils.text_utils import safe_convert_to_python_types, detect_language
from config.settings import (
    EDUCATIONAL_INAPPROPRIATE_THRESHOLD,
    EDUCATIONAL_OFFENSIVE_THRESHOLD,
    EDUCATIONAL_SEXUAL_THRESHOLD
)

def moderate_vietnamese_text(text: str) -> Dict[str, Any]:
    """Vietnamese text moderation using keyword-based approach"""
    results = {
        "toxicity_score": 0.0,
        "hate_speech_score": 0.0,
        "sexual_content_score": 0.0,
        "violence_score": 0.0,
        "language": "vietnamese",
        "method": "keyword_based"
    }
    
    text_lower = text.lower()
    
    # Vietnamese toxic/hate speech keywords
    vietnamese_toxic_keywords = [
        'đm', 'dm', 'đụ', 'địt', 'lồn', 'cặc', 'buồi', 'loz','fuck', 'óc chó'
    ]
    
    # Vietnamese sexual content keywords
    vietnamese_sexual_keywords = [
        'chịch', 'đụ', 'địt', 'lồn', 'cặc', 'buồi',
         'porn', 'phim sex', 'clip sex'
    ]
    
    # Vietnamese violence keywords
    vietnamese_violence_keywords = [
    ]
    
    # Count occurrences
    toxic_count = sum(1 for keyword in vietnamese_toxic_keywords if keyword in text_lower)
    sexual_count = sum(1 for keyword in vietnamese_sexual_keywords if keyword in text_lower)
    violence_count = sum(1 for keyword in vietnamese_violence_keywords if keyword in text_lower)
    
    # Calculate scores
    word_count = len(text.split())
    if word_count > 0:
        results["toxicity_score"] = min(toxic_count / word_count * 5, 1.0)
        results["sexual_content_score"] = min(sexual_count / word_count * 5, 1.0)
        results["violence_score"] = min(violence_count / word_count * 5, 1.0)
        results["hate_speech_score"] = results["toxicity_score"]
    
    return results

def moderate_text_advanced(text: str) -> Dict[str, Any]:
    """Advanced text moderation with Vietnamese support"""
    language = detect_language(text)
    
    results = {
        "toxicity_score": 0.0,
        "hate_speech_score": 0.0,
        "sexual_content_score": 0.0,
        "violence_score": 0.0,
        "overall_risk": "LOW",
        "language_detected": language,
        "details": {}
    }
    
    try:
        if language == "vietnamese":
            vietnamese_results = moderate_vietnamese_text(text)
            results.update(vietnamese_results)
            results["details"]["moderation_method"] = "vietnamese_keywords"
        else:
            # English AI models
            if model_manager.toxicity_pipeline:
                try:
                    toxicity_result = model_manager.toxicity_pipeline(text)
                    if isinstance(toxicity_result, list) and len(toxicity_result) > 0:
                        toxic_score = 0.0
                        for result in toxicity_result:
                            if result.get('label') in ['TOXIC', 'toxic', '1', 1]:
                                toxic_score = float(result.get('score', 0.0))
                                break
                            elif result.get('label') in ['CLEAN', 'clean', '0', 0]:
                                toxic_score = 1.0 - float(result.get('score', 0.0))
                                break
                        results["toxicity_score"] = toxic_score
                        results["details"]["toxicity_raw"] = toxicity_result
                except Exception as e:
                    print(f"Toxicity detection failed: {e}")
            
            if model_manager.hate_speech_pipeline:
                try:
                    hate_result = model_manager.hate_speech_pipeline(text)
                    if isinstance(hate_result, list) and len(hate_result) > 0:
                        hate_score = 0.0
                        for result in hate_result:
                            if result.get('label') in ['TOXIC', 'toxic', 'HATE', 'hate']:
                                hate_score = float(result.get('score', 0.0))
                                break
                        results["hate_speech_score"] = hate_score
                        results["details"]["hate_speech_raw"] = hate_result
                except Exception as e:
                    print(f"Hate speech detection failed: {e}")
            
            # English keyword fallback
            text_lower = text.lower()
            sexual_keywords = ['sex', 'sexual', 'nude', 'naked', 'porn', 'adult', 'explicit']
            violence_keywords = ['violence', 'violent', 'kill', 'murder', 'weapon', 'gun', 'blood', 'death']
            
            sexual_count = sum(1 for word in sexual_keywords if word in text_lower)
            violence_count = sum(1 for word in violence_keywords if word in text_lower)
            
            word_count = len(text.split())
            if word_count > 0:
                results["sexual_content_score"] = min(sexual_count / word_count * 3, 1.0)
                results["violence_score"] = min(violence_count / word_count * 3, 1.0)
            
            results["details"]["moderation_method"] = "english_ai_models"
        
        # Overall risk assessment
        max_score = max(
            results["toxicity_score"],
            results["hate_speech_score"], 
            results["sexual_content_score"],
            results["violence_score"]
        )
        
        if max_score > 0.7:
            results["overall_risk"] = "HIGH"
        elif max_score > 0.4:
            results["overall_risk"] = "MEDIUM"
        else:
            results["overall_risk"] = "LOW"
            
        results = safe_convert_to_python_types(results)
        return results
        
    except Exception as e:
        print(f"Text moderation processing failed: {e}")
        return {
            "toxicity_score": 0.0,
            "hate_speech_score": 0.0,
            "sexual_content_score": 0.0,
            "violence_score": 0.0,
            "overall_risk": "ERROR",
            "language_detected": language,
            "error": str(e)
        }

def moderate_educational_content(text: str) -> Dict[str, Any]:
    """Specialized moderation for educational content"""
    results = {
        "inappropriate_score": 0.0,
        "offensive_score": 0.0,
        "sexual_content_score": 0.0,
        "overall_risk": "LOW",
        "language_detected": detect_language(text),
        "moderation_method": "educational_specialized",
        "details": {}
    }
    
    try:
        language = detect_language(text)
        text_lower = text.lower()
        
        # Vietnamese educational content moderation
        if language == "vietnamese":
            # Vietnamese inappropriate words for educational context
            vietnamese_inappropriate = [
                'đm', 'dm', 'đụ', 'địt', 'lồn', 'cặc', 'buồi', 'loz', 'fuck',
                'óc chó', 'thằng ngu', 'con chó', 'đồ ngu', 'khốn nạn', 'đồ khốn',
                'chết tiệt', 'đồ chó', 'thằng chó', 'con chó', 'đồ súc vật'
            ]
            
            # Vietnamese sexual content
            vietnamese_sexual = [
                'chịch', 'đụ', 'địt', 'lồn', 'cặc', 'buồi', 'porn', 'phim sex',
                'clip sex', 'sex', 'tình dục', 'làm tình', 'quan hệ'
            ]
            
            # Count inappropriate words
            inappropriate_count = sum(1 for word in vietnamese_inappropriate if word in text_lower)
            sexual_count = sum(1 for word in vietnamese_sexual if word in text_lower)
            
            # Anti-dilution scoring: Score based on presence, not ratio
            if inappropriate_count > 0:
                # Base score for any inappropriate word
                base_score = 0.7
                # Additional score for multiple occurrences
                additional_score = min(inappropriate_count * 0.1, 0.3)
                results["inappropriate_score"] = min(base_score + additional_score, 1.0)
                results["offensive_score"] = results["inappropriate_score"]
            else:
                results["inappropriate_score"] = 0.0
                results["offensive_score"] = 0.0
            
            if sexual_count > 0:
                # Base score for any sexual word
                base_score = 0.6
                # Additional score for multiple occurrences
                additional_score = min(sexual_count * 0.1, 0.4)
                results["sexual_content_score"] = min(base_score + additional_score, 1.0)
            else:
                results["sexual_content_score"] = 0.0
            
            results["details"]["vietnamese_keywords_found"] = {
                "inappropriate": [word for word in vietnamese_inappropriate if word in text_lower],
                "sexual": [word for word in vietnamese_sexual if word in text_lower]
            }
            
        else:
            # English educational content moderation
            english_inappropriate = [
                'fuck', 'shit', 'damn', 'bitch', 'asshole', 'stupid', 'idiot',
                'moron', 'retard', 'gay', 'fag', 'nigger', 'whore', 'slut',
                'bastard', 'crap', 'piss', 'hell', 'bloody'
            ]
            
            english_sexual = [
                'sex', 'sexual', 'porn', 'pornography', 'nude', 'naked',
                'breast', 'penis', 'vagina', 'orgasm', 'masturbation'
            ]
            
            # Count inappropriate words
            inappropriate_count = sum(1 for word in english_inappropriate if word in text_lower)
            sexual_count = sum(1 for word in english_sexual if word in text_lower)
            
            # Anti-dilution scoring: Score based on presence, not ratio
            if inappropriate_count > 0:
                # Base score for any inappropriate word
                base_score = 0.6
                # Additional score for multiple occurrences
                additional_score = min(inappropriate_count * 0.1, 0.4)
                results["inappropriate_score"] = min(base_score + additional_score, 1.0)
                results["offensive_score"] = results["inappropriate_score"]
            else:
                results["inappropriate_score"] = 0.0
                results["offensive_score"] = 0.0
            
            if sexual_count > 0:
                # Base score for any sexual word
                base_score = 0.5
                # Additional score for multiple occurrences
                additional_score = min(sexual_count * 0.1, 0.5)
                results["sexual_content_score"] = min(base_score + additional_score, 1.0)
            else:
                results["sexual_content_score"] = 0.0
            
            results["details"]["english_keywords_found"] = {
                "inappropriate": [word for word in english_inappropriate if word in text_lower],
                "sexual": [word for word in english_sexual if word in text_lower]
            }
            
            # Use AI models for English if available
            if model_manager.educational_offensive_classifier:
                try:
                    offensive_result = model_manager.educational_offensive_classifier(text)
                    if isinstance(offensive_result, list) and len(offensive_result) > 0:
                        for result in offensive_result:
                            if result.get('label') in ['OFFENSIVE', 'offensive', '1', 1]:
                                ai_offensive_score = float(result.get('score', 0.0))
                                results["offensive_score"] = max(results["offensive_score"], ai_offensive_score)
                                break
                        results["details"]["ai_offensive_result"] = offensive_result
                except Exception as e:
                    print(f"Educational offensive classifier failed: {e}")
            
            if model_manager.educational_inappropriate_classifier:
                try:
                    inappropriate_result = model_manager.educational_inappropriate_classifier(text)
                    if isinstance(inappropriate_result, list) and len(inappropriate_result) > 0:
                        for result in inappropriate_result:
                            if result.get('label') in ['TOXIC', 'toxic', '1', 1]:
                                ai_inappropriate_score = float(result.get('score', 0.0))
                                results["inappropriate_score"] = max(results["inappropriate_score"], ai_inappropriate_score)
                                break
                        results["details"]["ai_inappropriate_result"] = inappropriate_result
                except Exception as e:
                    print(f"Educational inappropriate classifier failed: {e}")
        
        # Overall risk assessment for educational content
        max_score = max(
            results["inappropriate_score"],
            results["offensive_score"],
            results["sexual_content_score"]
        )
        
        if max_score > EDUCATIONAL_OFFENSIVE_THRESHOLD:
            results["overall_risk"] = "HIGH"
        elif max_score > EDUCATIONAL_INAPPROPRIATE_THRESHOLD:
            results["overall_risk"] = "MEDIUM"
        elif max_score > 0.2:
            results["overall_risk"] = "LOW-MEDIUM"
        else:
            results["overall_risk"] = "LOW"
        
        results = safe_convert_to_python_types(results)
        return results
        
    except Exception as e:
        print(f"Educational content moderation failed: {e}")
        return {
            "inappropriate_score": 0.0,
            "offensive_score": 0.0,
            "sexual_content_score": 0.0,
            "overall_risk": "ERROR",
            "language_detected": detect_language(text),
            "error": str(e)
        }
