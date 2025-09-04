#!/usr/bin/env python3
"""
Test script for Smart Chunking Performance
Compares old vs new chunking strategies
"""

import time
from services.file_processing import split_text_into_chunks, smart_text_chunking
from services.text_moderation import moderate_text_hybrid

def test_chunking_performance():
    """Test different chunking strategies"""
    
    print("🚀 Testing Smart Chunking Performance")
    print("=" * 60)
    
    # Test cases with different file sizes
    test_cases = [
        {
            "text": "This is a small document with only a few sentences.",
            "size": "Small",
            "expected_chunks_old": 1,
            "expected_chunks_new": 1
        },
        {
            "text": "This is a medium document. " * 100,  # ~2000 chars
            "size": "Medium", 
            "expected_chunks_old": 3,
            "expected_chunks_new": 1
        },
        {
            "text": "This is a large document. " * 500,  # ~10000 chars
            "size": "Large",
            "expected_chunks_old": 13,
            "expected_chunks_new": 4
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['size']} Document")
        print(f"Text length: {len(test_case['text'])} characters")
        
        # Test old chunking (800 chars)
        start_time = time.time()
        old_chunks = split_text_into_chunks(test_case['text'], chunk_size=800)
        old_time = time.time() - start_time
        
        # Test new smart chunking
        start_time = time.time()
        new_chunks = smart_text_chunking(test_case['text'])
        new_time = time.time() - start_time
        
        print(f"Old chunking (800 chars):")
        print(f"  - Chunks: {len(old_chunks)}")
        print(f"  - Time: {old_time:.4f}s")
        
        print(f"Smart chunking:")
        print(f"  - Chunks: {len(new_chunks)}")
        print(f"  - Time: {new_time:.4f}s")
        
        # Calculate performance improvement
        if old_time > 0:
            improvement = ((old_time - new_time) / old_time) * 100
            print(f"  - Improvement: {improvement:.1f}%")
        
        # Calculate AI calls reduction
        ai_calls_reduction = ((len(old_chunks) - len(new_chunks)) / len(old_chunks)) * 100
        print(f"  - AI calls reduction: {ai_calls_reduction:.1f}%")
        
        print("-" * 40)

def test_ai_model_performance():
    """Test AI model performance with different chunk sizes"""
    
    print("\n🤖 Testing AI Model Performance")
    print("=" * 60)
    
    # Test text
    test_text = "This is a test document with some content. " * 50  # ~2000 chars
    
    # Test as single chunk
    print("Single chunk processing:")
    start_time = time.time()
    result = moderate_text_hybrid(test_text)
    single_time = time.time() - start_time
    print(f"  - Time: {single_time:.4f}s")
    print(f"  - Method: {result.get('moderation_method', 'unknown')}")
    print(f"  - Risk: {result.get('overall_risk', 'unknown')}")
    
    # Test as multiple chunks
    print("\nMultiple chunks processing:")
    chunks = split_text_into_chunks(test_text, chunk_size=800)
    start_time = time.time()
    chunk_results = []
    for chunk in chunks:
        chunk_result = moderate_text_hybrid(chunk)
        chunk_results.append(chunk_result)
    multi_time = time.time() - start_time
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Time: {multi_time:.4f}s")
    print(f"  - Time per chunk: {multi_time/len(chunks):.4f}s")
    
    # Calculate overhead
    overhead = multi_time - single_time
    print(f"  - Overhead: {overhead:.4f}s")
    print(f"  - Overhead per chunk: {overhead/len(chunks):.4f}s")

if __name__ == "__main__":
    test_chunking_performance()
    test_ai_model_performance()
