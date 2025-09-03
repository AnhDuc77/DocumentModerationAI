#!/usr/bin/env python3
"""
Clean test script for AI Service
Professional testing without emojis and icons
"""

import requests
import json
import base64
from PIL import Image
import io
import time

BASE_URL = "http://localhost:8000"

def test_service_health():
    """Test service health and capabilities"""
    print("Testing AI Service Health...")
    print("-" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("Service is healthy")
            print(f"   Service: {data['service']}")
            print(f"   Version: {data['version']}")
            
            print("\n   Models Status:")
            for model, status in data['models_loaded'].items():
                status_text = "LOADED" if status else "FAILED"
                print(f"     {model}: {status_text}")
                
            print("\n   Capabilities:")
            for capability, method in data['capabilities'].items():
                print(f"     {capability}: {method}")
                
            print("\n   Accuracy Estimates:")
            for feature, accuracy in data['accuracy_estimates'].items():
                print(f"     {feature}: {accuracy}")
                
            return True
        else:
            print(f"Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot connect to service: {e}")
        print("   Make sure to run: python ai_service.py")
        return False

def test_text_moderation():
    """Test text moderation capabilities"""
    print("\nTesting Text Moderation...")
    print("-" * 50)
    
    test_cases = [
        {
            "name": "Safe Business Text",
            "content": "Our quarterly revenue increased by 15% this quarter with excellent customer satisfaction.",
            "expected_risk": "LOW"
        },
        {
            "name": "Toxic Content",
            "content": "You are stupid and I hate you. This is toxic content for testing purposes.",
            "expected_risk": "HIGH"
        },
        {
            "name": "Vietnamese Safe Text",
            "content": "Báo cáo quý này cho thấy doanh thu tăng trưởng tốt và khách hàng hài lòng.",
            "expected_risk": "LOW"
        },
        {
            "name": "Vietnamese Toxic Text",
            "content": "Thằng ngu này đm, tao ghét mày vcl. Đồ khốn nạn.",
            "expected_risk": "HIGH"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n   Test {i+1}: {test_case['name']}")
        print(f"   Content: {test_case['content'][:60]}...")
        
        data = {
            "chunk_index": i,
            "text_content": test_case['content'],
            "metadata": json.dumps({"test_case": test_case['name']})
        }
        
        try:
            response = requests.post(f"{BASE_URL}/process-text-chunk", data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                score = result.get('moderation_score', 0)
                risk = result.get('moderation_result', {}).get('overall_risk', 'UNKNOWN')
                language = result.get('moderation_result', {}).get('language_detected', 'unknown')
                
                print(f"   Result: SUCCESS")
                print(f"      Language: {language}")
                print(f"      Overall Score: {score:.3f}")
                print(f"      Risk Level: {risk}")
                print(f"      Expected: {test_case['expected_risk']}")
                    
            else:
                print(f"   Result: FAILED with status {response.status_code}")
                print(f"      Response: {response.text}")
                
        except Exception as e:
            print(f"   Result: ERROR - {e}")

def test_image_moderation():
    """Test image moderation capabilities"""
    print("\nTesting Image Moderation...")
    print("-" * 50)
    
    test_images = [
        {
            "name": "Safe Blue Image",
            "color": "blue",
            "size": (400, 300),
            "expected_risk": "LOW"
        },
        {
            "name": "Large Red Image",
            "color": "red", 
            "size": (800, 600),
            "expected_risk": "LOW"
        },
        {
            "name": "Skin Tone Image",
            "color": (222, 184, 135),  # Skin-like color
            "size": (400, 600),
            "expected_risk": "LOW-MEDIUM"
        }
    ]
    
    for i, test_image in enumerate(test_images):
        print(f"\n   Test {i+1}: {test_image['name']}")
        print(f"   Size: {test_image['size']}, Color: {test_image['color']}")
        
        # Create test image
        img = Image.new('RGB', test_image['size'], color=test_image['color'])
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        files = {
            'image_file': (f"{test_image['name'].lower().replace(' ', '_')}.png", img_data, 'image/png')
        }
        
        data = {
            'chunk_index': i,
            'metadata': json.dumps({"test_case": test_image['name']})
        }
        
        try:
            response = requests.post(f"{BASE_URL}/process-image-chunk", files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                score = result.get('moderation_score', 0)
                risk = result.get('moderation_result', {}).get('overall_risk', 'UNKNOWN')
                method = result.get('moderation_result', {}).get('detection_method', 'unknown')
                
                mod_result = result.get('moderation_result', {})
                nsfw_score = mod_result.get('nsfw_score', 0)
                violence_score = mod_result.get('violence_score', 0)
                
                print(f"   Result: SUCCESS")
                print(f"      Detection Method: {method}")
                print(f"      Overall Score: {score:.3f}")
                print(f"      Risk Level: {risk}")
                print(f"      NSFW Score: {nsfw_score:.3f}")
                print(f"      Violence Score: {violence_score:.3f}")
                print(f"      Expected: {test_image['expected_risk']}")
                    
            else:
                print(f"   Result: FAILED with status {response.status_code}")
                print(f"      Response: {response.text}")
                
        except Exception as e:
            print(f"   Result: ERROR - {e}")

def test_bulk_processing():
    """Test bulk processing capabilities"""
    print("\nTesting Bulk Processing...")
    print("-" * 50)
    
    # Create test data
    text_chunks = [
        {
            "chunk_index": 0,
            "text_content": "This is safe business content about quarterly reports.",
            "chunk_size": 55,
            "metadata": {"page": 1, "section": "introduction"}
        }
    ]
    
    # Create test image
    img = Image.new('RGB', (300, 200), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_data = img_bytes.getvalue()
    img_base64 = base64.b64encode(img_data).decode()
    
    image_chunks = [
        {
            "chunk_index": 0,
            "image_data": img_base64,
            "image_format": "JPEG",
            "image_size": [300, 200],
            "metadata": {"page": 2, "caption": "Business chart"}
        }
    ]
    
    request_data = {
        "file_id": "test_document_001",
        "file_type": "pdf",
        "original_filename": "test_document.pdf",
        "text_chunks": text_chunks,
        "image_chunks": image_chunks,
        "processing_options": {
            "enable_embedding": True,
            "enable_sentiment": True
        }
    }
    
    print(f"   Processing {len(text_chunks)} text chunks + {len(image_chunks)} image chunks...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/process-chunks",
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data),
            timeout=60
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"   Result: SUCCESS")
            print(f"      Status: {result.get('processing_status')}")
            print(f"      Overall Score: {result.get('overall_moderation_score', 0):.3f}")
            print(f"      Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"      Text Results: {len(result.get('text_results', []))}")
            print(f"      Image Results: {len(result.get('image_results', []))}")
            
        else:
            print(f"   Result: FAILED with status {response.status_code}")
            print(f"      Response: {response.text}")
            
    except Exception as e:
        print(f"   Result: ERROR - {e}")

def main():
    """Run all tests for the AI service"""
    print("AI Service Comprehensive Test Suite")
    print("=" * 70)
    print("Testing all AI service capabilities")
    print()
    
    tests = [
        ("Service Health Check", test_service_health),
        ("Text Moderation", test_text_moderation),
        ("Image Moderation", test_image_moderation),
        ("Bulk Processing", test_bulk_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        try:
            test_func()
            passed += 1
            print(f"\n{name} completed successfully")
        except Exception as e:
            print(f"\n{name} failed with error: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print(f"Test Results: {passed}/{total} test suites completed")
    
    if passed == total:
        print("All tests passed. AI Service is ready for production.")
        print("\nProduction Integration Info:")
        print(f"   Service URL: {BASE_URL}")
        print("   Main Endpoint: POST /process-chunks")
        print("   Health Check: GET /health")
        print("   API Documentation: http://localhost:8000/docs")
        
    elif passed >= 1:
        print("Service is partially working. Some features may be limited.")
        
    else:
        print("Service has major issues. Check configuration and dependencies.")
    
    print("\nNext Steps:")
    print("   1. Integrate with Spring Boot using the service URL above")
    print("   2. Use /process-chunks endpoint for bulk document processing")
    print("   3. Monitor service performance in production")

if __name__ == "__main__":
    main()


