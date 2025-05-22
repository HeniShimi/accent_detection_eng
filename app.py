import streamlit as st
import subprocess
import os
import re
from speechbrain.pretrained import EncoderClassifier

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="English Accent Detector", layout="centered")

# --- LOAD ACCENT DETECTOR MODEL FROM SPEECHBRAIN ---
@st.cache_resource
def load_accent_detector():
    try:
        st.info("Loading SpeechBrain accent model...")
        model = EncoderClassifier.from_hparams(
            source="speechbrain/accent-id-commonaccent_ecapa",
            savedir="tmpdir_accent_model"
        )
        st.success("Accent model loaded successfully!")
        return model
    except Exception as e:
        st.error("Failed to load SpeechBrain accent model.")
        st.exception(e)
        return None

accent_detector = load_accent_detector()

# --- AUDIO EXTRACTION FUNCTION ---
def extract_audio_from_video(video_url, output_audio_path="audio.wav"):
    try:
        command = [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "-o", output_audio_path,
            video_url
        ]
        st.info(f"Downloading and extracting audio from: {video_url}")
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        st.success(f"Audio extracted successfully to: {output_audio_path}")
        if process.stdout:
            st.code(process.stdout)
        if process.stderr:
            st.error(process.stderr)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error extracting audio: {e.stderr}")
        return False
    except Exception as e:
        st.error(f"Unexpected error during audio extraction: {e}")
        return False

# --- ACCENT ANALYSIS FUNCTION ---
def analyze_accent(audio_path):
    if accent_detector is None:
        st.error("Accent detector not loaded.")
        return None, None, None

    try:
        st.info(f"Analyzing accent in: {audio_path}")
        out_prob, score, index, text_lab = accent_detector.classify_file(audio_path)

        accent = text_lab
        confidence = score.item() * 100  # Convert to percentage
        summary = f"The detected accent is **{accent.replace('_', ' ').title()}**."

        if accent == "american_english":
            summary += " This indicates a North American English pronunciation style."
        elif accent == "british_english":
            summary += " This suggests a British English pronunciation style."
        elif accent == "australian_english":
            summary += " This implies an Australian English pronunciation style."
        elif accent == "indian_english":
            summary += " This points to an Indian English pronunciation style."
        elif accent == "non_native_english":
            summary += " The accent is identified as non-native English, indicating a significant influence from another language."
        else:
            summary += " Further linguistic analysis may be needed for more specific details."

        return accent, confidence, summary
    except Exception as e:
        st.error(f"Error analyzing accent: {e}")
        return None, None, None

# --- STREAMLIT UI ---
st.title("🗣️ English Accent Detector for Hiring")
st.markdown("""
This tool helps evaluate spoken English accents from public video URLs.
It extracts audio, analyzes the accent, and provides a classification and confidence score.
""")

video_url = st.text_input("Enter Public Video URL (e.g., Loom, direct MP4 link):")

if video_url:
    if not re.match(r"^(http|https)://[^\s/$.?#].[^\s]*$", video_url):
        st.error("Please enter a valid URL.")
    else:
        audio_output_path = "extracted_audio.wav"
        if st.button("Analyze Accent"):
            st.session_state['analysis_started'] = True
            st.session_state['audio_extracted'] = False
            st.session_state['accent_analyzed'] = False
            st.session_state['accent'] = None
            st.session_state['confidence'] = None
            st.session_state['summary'] = None

if st.session_state.get('analysis_started'):
    audio_output_path = "extracted_audio.wav"
    if not st.session_state.get('audio_extracted'):
        with st.spinner("Extracting audio... This may take a moment."):
            if extract_audio_from_video(video_url, audio_output_path):
                st.session_state['audio_extracted'] = True
            else:
                st.session_state['analysis_started'] = False

    if st.session_state.get('audio_extracted') and not st.session_state.get('accent_analyzed'):
        if accent_detector is None:
            st.error("Cannot proceed with analysis: Model failed to load during startup.")
            st.session_state['analysis_started'] = False
        else:
            with st.spinner("Analyzing accent..."):
                accent, confidence, summary = analyze_accent(audio_output_path)
                st.session_state['accent'] = accent
                st.session_state['confidence'] = confidence
                st.session_state['summary'] = summary
                st.session_state['accent_analyzed'] = True
                if os.path.exists(audio_output_path):
                    os.remove(audio_output_path)

    if st.session_state.get('accent_analyzed'):
        if st.session_state['accent'] is not None:
            st.subheader("Analysis Results:")
            st.write(f"**Detected Accent:** `{st.session_state['accent'].replace('_', ' ').title()}`")
            st.write(f"**Confidence in English Accent Score:** `{st.session_state['confidence']:.2f}%`")
            st.markdown(f"**Summary:** {st.session_state['summary']}")
        else:
            st.warning("Could not analyze accent. Please try a different video or check the URL.")
