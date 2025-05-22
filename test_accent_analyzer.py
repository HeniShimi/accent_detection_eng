"""
Test script for the English accent analyzer with Hugging Face model.
This script allows testing the complete pipeline with different videos.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video_processor import VideoProcessor
from accent_analyzer import AccentAnalyzer

def test_video_url(url):
    """
    Test accent analysis on a specific video URL.
    
    Args:
        url (str): URL of the video to test.
        
    Returns:
        dict: Analysis results.
    """
    print(f"\n=== Testing URL: {url} ===")
    
    try:
        # Initialize processors
        video_processor = VideoProcessor()
        accent_analyzer = AccentAnalyzer()
        
        # Step 1: Download video and extract audio
        print("Downloading video and extracting audio...")
        audio_path = video_processor.process_video_url(url)
        print(f"Audio extracted: {audio_path}")
        
        # Step 2: Analyze accent
        print("Analyzing accent...")
        results = accent_analyzer.analyze_audio(audio_path)
        
        # Display results
        print("\nAnalysis Results:")
        print(json.dumps(results, indent=2))
        
        # Clean up temporary files
        video_processor.cleanup(audio_path)
        
        return results
    
    except Exception as e:
        print(f"\nError during test: {e}")
        return {"error": str(e)}

def run_test_suite():
    """
    Run a test suite on different videos with different accents.
    """
    # List of test videos with different accents
    test_videos = [
        {
            "url": "https://www.youtube.com/watch?v=3FtGOHUkEzI",
            "description": "American accent (US)",
            "expected_accent": "American"
        },
        {
            "url": "https://www.youtube.com/watch?v=qYlmFISLO9M",
            "description": "British accent (UK)",
            "expected_accent": "British"
        },
        {
            "url": "https://www.youtube.com/watch?v=4LvWYP7839Q",
            "description": "Australian accent",
            "expected_accent": "Australian"
        },
        {
            "url": "https://www.youtube.com/watch?v=QYlVJlmjLEc",
            "description": "Indian accent",
            "expected_accent": "Indian"
        },
        {
            "url": "https://download.ted.com/talks/KateDarling_2018S.mp4",
            "description": "TED Talk (Direct MP4)",
            "expected_accent": "American"
        }
    ]
    
    results = []
    
    print("\n=== Starting test suite ===\n")
    
    for i, test in enumerate(test_videos, 1):
        print(f"\nTest {i}/{len(test_videos)}: {test['description']}")
        result = test_video_url(test["url"])
        
        # Check if result matches expectation
        expected = test["expected_accent"]
        actual = result.get("accent", "").split()[0] if "accent" in result else "Error"
        
        success = expected.lower() in actual.lower()
        status = "✅ SUCCESS" if success else "❌ FAILURE"
        
        print(f"\nResult: {status}")
        print(f"Expected accent: {expected}")
        print(f"Detected accent: {actual}")
        
        results.append({
            "test": test["description"],
            "url": test["url"],
            "expected": expected,
            "actual": actual,
            "success": success,
            "details": result
        })
    
    print("\n=== Test Summary ===")
    success_count = sum(1 for r in results if r["success"])
    print(f"Tests passed: {success_count}/{len(test_videos)} ({success_count/len(test_videos)*100:.1f}%)")
    
    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        print(f"{status} Test {i}: {result['test']} - Expected: {result['expected']}, Got: {result['actual']}")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the English accent analyzer with Hugging Face model")
    parser.add_argument("--url", help="URL of the video to test")
    parser.add_argument("--suite", action="store_true", help="Run the complete test suite")
    
    args = parser.parse_args()
    
    # Run the test
    if args.suite:
        run_test_suite()
    elif args.url:
        test_video_url(args.url)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

