"""
Test script to verify all API endpoints work correctly
Run this AFTER starting the Flask server with: python app.py
"""

import requests
import json
import base64
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:5000"

print("=" * 60)
print("TESTING KARATE ML API ENDPOINTS")
print("=" * 60)

# TEST 1: Health Check (GET)
print("\n1. Testing Health Check Endpoint...")
print(f"   GET {BASE_URL}/health")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"   ✓ Status Code: {response.status_code}")
    print(f"   ✓ Response: {response.json()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# TEST 2: Analyze Video with File Upload (POST)
print("\n2. Testing Video Analysis with File Upload...")
print(f"   POST {BASE_URL}/api/analyze")

# Check if test video exists
test_video = Path("nika3_perfect_reference.mp4")
if test_video.exists():
    try:
        with open(test_video, 'rb') as f:
            files = {'video': ('test.mp4', f, 'video/mp4')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files)
            print(f"   ✓ Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Video ID: {result.get('video_id')}")
                print(f"   ✓ Score: {result.get('analysis', {}).get('score')}")
                print(f"   ✓ Grade: {result.get('analysis', {}).get('grade')}")
                video_id = result.get('video_id')
            else:
                print(f"   ✗ Error: {response.text}")
                video_id = None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        video_id = None
else:
    print(f"   ⊘ Skipped - test video not found")
    video_id = None

# TEST 3: Analyze Video with URL (POST)
print("\n3. Testing Video Analysis with URL...")
print(f"   POST {BASE_URL}/api/analyze")
print("   ⊘ Skipped - requires Google Cloud Storage URL")

# TEST 4: Get Analyzed Video (GET)
if video_id:
    print("\n4. Testing Get Analyzed Video...")
    print(f"   GET {BASE_URL}/api/video/{video_id}")
    try:
        response = requests.get(f"{BASE_URL}/api/video/{video_id}")
        print(f"   ✓ Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✓ Content-Type: {response.headers.get('Content-Type')}")
            print(f"   ✓ Video Size: {len(response.content)} bytes")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print("\n4. Testing Get Analyzed Video...")
    print("   ⊘ Skipped - no video_id from previous test")

# TEST 5: Get Analyzed Video as Base64 (GET)
if video_id:
    print("\n5. Testing Get Video as Base64...")
    print(f"   GET {BASE_URL}/api/video/{video_id}?format=base64")
    try:
        response = requests.get(f"{BASE_URL}/api/video/{video_id}?format=base64")
        print(f"   ✓ Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Base64 length: {len(result.get('base64', ''))} chars")
            print(f"   ✓ Size: {result.get('size_bytes')} bytes")
            print(f"   ✓ MIME type: {result.get('mime_type')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print("\n5. Testing Get Video as Base64...")
    print("   ⊘ Skipped - no video_id from previous test")

# TEST 6: Get Analysis JSON (GET)
if video_id:
    print("\n6. Testing Get Analysis JSON...")
    print(f"   GET {BASE_URL}/api/analysis/{video_id}")
    try:
        response = requests.get(f"{BASE_URL}/api/analysis/{video_id}")
        print(f"   ✓ Status Code: {response.status_code}")
        if response.status_code == 200:
            analysis = response.json()
            print(f"   ✓ Score: {analysis.get('score')}")
            print(f"   ✓ Grade: {analysis.get('grade')}")
            print(f"   ✓ Body parts analyzed: {len(analysis.get('body_parts', {}))}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print("\n6. Testing Get Analysis JSON...")
    print("   ⊘ Skipped - no video_id from previous test")

# TEST 7: Analyze with Video Buffer in Response (POST)
print("\n7. Testing Video Analysis with Buffer Return...")
print(f"   POST {BASE_URL}/api/analyze?include_video=true")
if test_video.exists():
    try:
        with open(test_video, 'rb') as f:
            files = {'video': ('test.mp4', f, 'video/mp4')}
            response = requests.post(f"{BASE_URL}/api/analyze?include_video=true", files=files)
            print(f"   ✓ Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Video ID: {result.get('video_id')}")
                print(f"   ✓ Score: {result.get('analysis', {}).get('score')}")
                has_video = 'video_data' in result
                print(f"   ✓ Video buffer included: {has_video}")
                if has_video:
                    print(f"   ✓ Buffer size: {result['video_data'].get('size_bytes')} bytes")
            else:
                print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print(f"   ⊘ Skipped - test video not found")

# TEST 8: Cleanup Files (DELETE)
if video_id:
    print("\n8. Testing Cleanup Endpoint...")
    print(f"   DELETE {BASE_URL}/api/cleanup/{video_id}")
    try:
        response = requests.delete(f"{BASE_URL}/api/cleanup/{video_id}")
        print(f"   ✓ Status Code: {response.status_code}")
        print(f"   ✓ Response: {response.json()}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print("\n8. Testing Cleanup Endpoint...")
    print("   ⊘ Skipped - no video_id from previous test")

print("\n" + "=" * 60)
print("TESTING COMPLETE")
print("=" * 60)
print("\nNOTE: To test with external devices on WiFi:")
print("1. Find your IP: ipconfig (look for IPv4 Address)")
print("2. Replace localhost with your IP (e.g., 192.168.1.100)")
print("3. Access from other device: http://YOUR_IP:5000/health")
print("=" * 60)
