"""
Test suite for the refactored AI Document Processing Service
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:9000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Service: {data.get('service', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
            print(f"   Models loaded: {data.get('models_loaded', {})}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_text_moderation():
    """Test text moderation endpoint"""
    print("\nTesting text moderation...")
    try:
        test_text = "This is a test message for moderation."
        response = requests.post(f"{BASE_URL}/moderate-text", data=test_text)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Text moderation passed")
            print(f"   Overall risk: {data.get('overall_risk', 'Unknown')}")
            print(f"   Language: {data.get('language_detected', 'Unknown')}")
            return True
        else:
            print(f"❌ Text moderation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Text moderation error: {e}")
        return False

def test_educational_moderation():
    """Test educational content moderation"""
    print("\nTesting educational content moderation...")
    try:
        test_text = "This is educational content for testing."
        response = requests.post(f"{BASE_URL}/moderate-educational-text", data=test_text)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Educational moderation passed")
            print(f"   Overall risk: {data.get('overall_risk', 'Unknown')}")
            print(f"   Inappropriate score: {data.get('inappropriate_score', 0.0)}")
            return True
        else:
            print(f"❌ Educational moderation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Educational moderation error: {e}")
        return False

def test_vietnamese_moderation():
    """Test Vietnamese text moderation"""
    print("\nTesting Vietnamese text moderation...")
    try:
        test_text = "Đây là nội dung tiếng Việt để kiểm tra."
        response = requests.post(f"{BASE_URL}/moderate-educational-text", data=test_text)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Vietnamese moderation passed")
            print(f"   Language detected: {data.get('language_detected', 'Unknown')}")
            print(f"   Overall risk: {data.get('overall_risk', 'Unknown')}")
            return True
        else:
            print(f"❌ Vietnamese moderation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Vietnamese moderation error: {e}")
        return False

def test_hash_computation():
    """Test hash computation utility"""
    print("\nTesting hash computation...")
    try:
        test_text = "Test text for hash computation"
        response = requests.post(f"{BASE_URL}/compute-hashes", data=test_text)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Hash computation passed")
            print(f"   Text hash: {data.get('text_hash', 'Unknown')[:20]}...")
            print(f"   SimHash: {data.get('simhash', 'Unknown')[:20]}...")
            return True
        else:
            print(f"❌ Hash computation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Hash computation error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting AI Document Processing Service Tests")
    print("=" * 50)
    
    tests = [
        test_health_check,
        test_text_moderation,
        test_educational_moderation,
        test_vietnamese_moderation,
        test_hash_computation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the service status.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()
