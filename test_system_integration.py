#!/usr/bin/env python3
"""
Test script to verify the complete system integration
"""

import requests
import json
import time

def test_system_integration():
    """Test the complete system integration"""
    
    print("🧪 Testing Pragati AI System Integration")
    print("=" * 50)
    
    # Test idea
    idea_name = "Smart Home Automation Platform"
    idea_concept = "AI-powered smart home platform that learns user preferences and automates daily routines. Includes voice control, energy optimization, and security features for Indian households."
    
    print(f"📝 Testing Idea: {idea_name}")
    print(f"📋 Concept: {idea_concept}")
    print()
    
    try:
        # Test validation endpoint
        print("🚀 Testing validation endpoint...")
        response = requests.post('http://localhost:5000/api/validate', 
                               json={
                                   'idea_name': idea_name,
                                   'idea_concept': idea_concept
                               },
                               timeout=300)  # 5 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Validation successful!")
            print(f"📊 Overall Score: {result['overall_score']}/5.0")
            print(f"🏆 Outcome: {result['validation_outcome']}")
            print(f"🤖 Agents Consulted: {result['total_agents_consulted']}")
            print(f"⏱️ Processing Time: {result['total_processing_time']:.2f} seconds")
            print()
            
            # Test PDF generation
            print("📄 Testing PDF generation...")
            pdf_response = requests.post('http://localhost:5000/api/generate-pdf',
                                       json={
                                           'idea_name': idea_name,
                                           'idea_concept': idea_concept
                                       },
                                       timeout=300)
            
            if pdf_response.status_code == 200:
                pdf_result = pdf_response.json()
                print("✅ PDF generation successful!")
                print(f"📁 PDF URL: {pdf_result['pdf_url']}")
                print(f"📄 Filename: {pdf_result['filename']}")
                print()
                
                # Test PDF download
                print("⬇️ Testing PDF download...")
                download_url = f"http://localhost:5000{pdf_result['pdf_url']}"
                download_response = requests.get(download_url)
                
                if download_response.status_code == 200:
                    print("✅ PDF download successful!")
                    print(f"📊 PDF size: {len(download_response.content)} bytes")
                else:
                    print(f"❌ PDF download failed: {download_response.status_code}")
            else:
                print(f"❌ PDF generation failed: {pdf_response.status_code}")
                print(f"Error: {pdf_response.text}")
        else:
            print(f"❌ Validation failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Wait a moment for the server to start
    print("⏳ Waiting for server to start...")
    time.sleep(5)
    
    test_system_integration()
