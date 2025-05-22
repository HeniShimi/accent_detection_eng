"""
Streamlit application for English accent analysis from online videos.
"""

import os
import sys
import tempfile
import streamlit as st
import time
from pathlib import Path

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video_processor import VideoProcessor
from accent_analyzer import AccentAnalyzer

# Streamlit page configuration
st.set_page_config(
    page_title="English Accent Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .accent-label {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 10px;
    }
    .confidence-meter {
        margin: 20px 0;
    }
    .explanation-text {
        font-size: 1rem;
        line-height: 1.5;
        color: #424242;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #9e9e9e;
        font-size: 0.8rem;
    }
    .error-message {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def create_temp_dir():
    """Create a temporary directory for application files."""
    temp_dir = os.path.join(tempfile.gettempdir(), "accent_analyzer")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def display_header():
    """Display the application header."""
    st.markdown('<h1 class="main-header">English Accent Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze English accents from online videos</p>', unsafe_allow_html=True)

    st.markdown("""
    This tool allows you to analyze a speaker's English accent from an online video.
    It detects the type of accent (American, British, Australian, etc.) and provides a confidence score.

    **Instructions:**
    1. Paste a public video URL (YouTube, Loom, direct MP4, etc.)
    2. Click "Analyze Accent"
    3. Wait for processing (this may take a moment)
    4. View the analysis results
    """)

def display_url_input():
    """Display the URL input form."""
    with st.form(key="url_form"):
        url = st.text_input(
            "Video URL",
            placeholder="https://www.youtube.com/watch?v=example",
            help="Enter a public video URL (YouTube, Loom, etc.)"
        )
        submit_button = st.form_submit_button(label="Analyze Accent")
    return url, submit_button

def display_processing_status():
    """Display processing status with a progress bar."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_steps = [
        ("Downloading video...", 20),
        ("Extracting audio...", 40),
        ("Transcribing audio...", 60),
        ("Analyzing accent...", 80),
        ("Finalizing results...", 100),
    ]
    for text, progress in status_steps:
        status_text.text(text)
        progress_bar.progress(progress)
        time.sleep(0.5)

    status_text.empty()
    progress_bar.empty()

def display_results(results):
    """Display accent analysis results."""
    st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if "error" in results:
        st.markdown(f'<div class="error-message">{results["message"]}</div>', unsafe_allow_html=True)
        st.error(f"Error details: {results['error']}")
        return

    if not results.get("is_english", True):
        st.markdown(
            f'<div class="error-message">The detected language is not English. '
            f'Detected language: {results.get("detected_language", "unknown")}</div>',
            unsafe_allow_html=True
        )
        return

    st.markdown(f'<div class="accent-label">Detected Accent: {results["accent"]}</div>', unsafe_allow_html=True)
    confidence = results["confidence"]
    st.markdown('<div class="confidence-meter">', unsafe_allow_html=True)
    st.markdown(f"**Confidence Score:** {confidence}%")
    st.progress(confidence / 100)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="explanation-text">', unsafe_allow_html=True)
    st.markdown("**Explanation:**")
    st.markdown(results["explanation"])
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("View Transcription"):
        st.markdown(results["transcription"])

    st.markdown('</div>', unsafe_allow_html=True)

def process_video_url(url, temp_dir):
    """Process video URL and analyze accent."""
    try:
        video_processor = VideoProcessor(temp_dir=temp_dir)
        accent_analyzer = AccentAnalyzer(openai_api_key=openai_api_key)

        with st.spinner("Downloading video and extracting audio..."):
            audio_path = video_processor.process_video_url(url)

        with st.spinner("Analyzing accent..."):
            results = accent_analyzer.analyze_audio(audio_path)

        video_processor.cleanup(audio_path)
        return results

    except Exception as e:
        return {
            "error": str(e),
            "message": "An error occurred while processing the video."
        }

def display_examples():
    """Display example URLs for testing."""
    with st.expander("Example URLs for testing"):
        st.markdown("""
        Here are some example URLs you can use to test the application:

        - **American accent**: https://www.youtube.com/watch?v=3FtGOHUkEzI  
        - **British accent**: https://www.youtube.com/watch?v=qYlmFISLO9M  
        - **Australian accent**: https://www.youtube.com/watch?v=4LvWYP7839Q  
        - **Indian accent**: https://www.youtube.com/watch?v=QYlVJlmjLEc

        *Note: These are public example videos available on YouTube.*
        """)

def display_footer():
    """Display the application footer."""
    st.markdown('<div class="footer">English Accent Analyzer - Developed By Heni S'himi - 2025</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application function."""
    temp_dir = create_temp_dir()
    display_header()
    url, submit_button = display_url_input()
    display_examples()

    if submit_button and url:
        if not url.startswith(("http://", "https://")):
            st.error("Please enter a valid URL starting with http:// or https://")
        else:
            display_processing_status()
            results = process_video_url(url, temp_dir)
            display_results(results)

    display_footer()

if __name__ == "__main__":
    main()

