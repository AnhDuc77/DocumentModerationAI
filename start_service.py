#!/usr/bin/env python3
"""
Startup script for AI Service
Handles installation and service startup
"""

import subprocess
import sys
import time
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        return False

def check_models():
    """Check if models can be loaded"""
    print("Checking AI models...")
    try:
        # Test basic imports
        import torch
        from transformers import pipeline
        from PIL import Image
        import nudenet
        
        print("Core libraries available")
        
        # Test Falconsai model
        print("   Testing Falconsai model...")
        classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
        print("   Falconsai NSFW model loaded successfully")
        
        return True
    except Exception as e:
        print(f"Warning: Model check failed: {e}")
        print("   Models will be downloaded on first run")
        return True  # Continue anyway

def start_service():
    """Start the AI service"""
    print("Starting AI Service...")
    try:
        # Import and run the service
        import ai_service
        print("AI Service started successfully")
    except Exception as e:
        print(f"Failed to start service: {e}")
        return False

def main():
    """Main startup routine"""
    print("AI Document Moderation Service Startup")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Installation failed. Please check your Python environment.")
        return
    
    # Step 2: Check models
    check_models()
    
    # Step 3: Start service
    print("\nStarting service...")
    print("   Service URL: http://localhost:8888")
    print("   API Documentation: http://localhost:8888/docs")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        # Run the service directly
        os.system("python ai_service.py")
    except KeyboardInterrupt:
        print("\nService stopped by user")
    except Exception as e:
        print(f"\nService error: {e}")

if __name__ == "__main__":
    main()
