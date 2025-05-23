import streamlit as st
import whisper
import os
import numpy as np
import librosa
import tempfile
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import torch
import soundfile as sf
import torchaudio
import yt_dlp
from pydub import AudioSegment
import matplotlib.pyplot as plt
from io import BytesIO

# Initialize models
@st.cache_resource
def load_models():
    # Whisper for transcription
    whisper_model = whisper.load_model("base")
    
    # Wav2Vec2 for accent detection
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    accent_model = AutoModelForAudioClassification.from_pretrained("TalTechNLP/voxlingua107-accent")
    
    return whisper_model, feature_extractor, accent_model

# Accent label mapping (based on VoxLingua107 model)
ACCENT_LABELS = {
    0: "American",
    1: "British",
    2: "Australian",
    3: "Canadian",
    4: "Irish",
    5: "Indian",
    6: "Scottish",
    7: "South African",
    8: "New Zealand",
    9: "Other English"
}

def download_and_extract_audio(video_url, temp_dir):
    audio_path = os.path.join(temp_dir, "audio.wav")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(temp_dir, 'audio'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # Ensure the file exists
        if not os.path.exists(audio_path):
            # Try alternative extension
            audio_path = os.path.join(temp_dir, "audio.m4a")
            if not os.path.exists(audio_path):
                raise FileNotFoundError("Audio extraction failed")
        
        # Convert to WAV if needed
        if not audio_path.endswith('.wav'):
            sound = AudioSegment.from_file(audio_path)
            audio_path = os.path.join(temp_dir, "audio.wav")
            sound.export(audio_path, format="wav")
        
        return audio_path
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

def preprocess_audio(audio_path):
    try:
        # Load audio file and resample to 16kHz if needed
        speech, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(speech)
        
        # Convert to mono if stereo
        if speech.shape[0] > 1:
            speech = torch.mean(speech, dim=0, keepdim=True)
        
        # Trim silence
        speech = speech.squeeze().numpy()
        speech, _ = librosa.effects.trim(speech, top_db=30)
        
        return np.expand_dims(speech, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing audio: {str(e)}")
        return None

def analyze_accent(audio_path, feature_extractor, accent_model):
    try:
        # Preprocess audio
        speech = preprocess_audio(audio_path)
        if speech is None:
            return "Error", 0, []
        
        # Extract features
        inputs = feature_extractor(
            speech[0],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Predict accent
        with torch.no_grad():
            logits = accent_model(**inputs).logits
        
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_label = torch.max(probs, dim=1)
        
        # Get confidence score
        confidence = top_prob.item() * 100
        
        # Get accent label
        accent = ACCENT_LABELS.get(top_label.item(), "Unknown")
        
        # Get top 3 accents with probabilities
        top_probs, top_labels = torch.topk(probs, 3)
        details = [
            (ACCENT_LABELS.get(label.item(), "Unknown"), prob.item() * 100)
            for prob, label in zip(top_probs[0], top_labels[0])
        ]
        
        return accent, confidence, details
        
    except Exception as e:
        st.error(f"Error in accent analysis: {str(e)}")
        return "Error", 0, []

def transcribe_audio(model, audio_path):
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return "Transcription failed"

def get_accent_explanation(accent):
    explanations = {
        "American": "Characterized by rhotic pronunciation (pronouncing 'r' after vowels), flatter vowel sounds, and specific patterns like the cot-caught merger.",
        "British": "Typically non-rhotic (dropping 'r' after vowels), with more rounded vowel sounds and distinct patterns like the trap-bath split.",
        "Australian": "Features vowel shifts (like raising of /æ/), non-rhotic, with distinctive intonation patterns.",
        "Canadian": "Similar to American but with Canadian raising (different pronunciation of diphthongs before voiceless consonants).",
        "Irish": "Rhotic with distinctive vowel qualities and consonant patterns (like slender and broad consonants).",
        "Indian": "Often syllable-timed rather than stress-timed, with retroflex consonants and distinctive vowel realizations.",
        "Scottish": "Rhotic with preserved Scots features, distinct vowel system (like no foot-strut split).",
        "South African": "Non-rhotic with raised /æ/, distinctive /ɒ/ realization, and influence from Afrikaans.",
        "New Zealand": "Non-rhotic with the 'ear-air' merger and distinctive vowel shifts (like centralized /ɪ/).",
        "Other English": "This accent shows typical English pronunciation patterns with some regional variations."
    }
    return explanations.get(accent, "This accent shows typical English pronunciation patterns with some regional variations.")

def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='blue')
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    return fig

def plot_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title('MFCC Features')
    return fig

def main():
    st.set_page_config(page_title="REM Waste Accent Analyzer", layout="wide")
    
    st.title("🎙️ REM Waste English Accent Detection")
    st.write("""
    Analyze English accents in video recordings for hiring evaluation purposes.
    Supports YouTube, Loom, and direct video links.
    """)
    
    with st.expander("ℹ️ How to use this tool"):
        st.write("""
        1. Paste a public video URL (YouTube, Loom, or direct MP4 link)
        2. Click "Analyze Accent"
        3. View the detailed accent analysis and transcription
        """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        video_url = st.text_input("Enter video URL:", placeholder="https://www.youtube.com/watch?v=... or https://www.loom.com/share/...")
    
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("Analyze Accent", type="primary")
    
    if analyze_btn and video_url:
        with st.spinner("Downloading and processing video..."):
            try:
                # Create temp directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Load models
                    whisper_model, feature_extractor, accent_model = load_models()
                    
                    # Download and extract audio
                    audio_path = download_and_extract_audio(video_url, temp_dir)
                    if not audio_path:
                        st.error("Failed to download and extract audio from the video URL")
                        return
                    
                    # Verify audio file
                    if not os.path.exists(audio_path):
                        st.error("Audio file not found after download")
                        return
                    
                    # Display audio player
                    st.audio(audio_path, format="audio/wav")
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3 = st.tabs(["Accent Analysis", "Speech Transcription", "Technical Details"])
                    
                    with tab1:
                        # Analyze accent
                        accent, confidence, accent_details = analyze_accent(
                            audio_path, feature_extractor, accent_model
                        )
                        
                        if accent == "Error":
                            st.error("Accent analysis failed")
                            return
                        
                        # Display results in columns
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.subheader("Primary Accent Detection")
                            st.metric("Detected Accent", accent)
                            st.metric("Confidence Score", f"{confidence:.1f}%")
                            
                            # Display explanation
                            st.subheader("Accent Characteristics")
                            st.info(get_accent_explanation(accent))
                        
                        with col_b:
                            st.subheader("Alternative Possibilities")
                            for alt_accent, alt_prob in accent_details[1:]:
                                st.write(f"**{alt_accent}**: {alt_prob:.1f}% confidence")
                                st.progress(int(alt_prob))
                        
                        # Visualizations
                        st.subheader("Audio Analysis")
                        col_v1, col_v2 = st.columns(2)
                        
                        with col_v1:
                            st.pyplot(plot_waveform(audio_path))
                        
                        with col_v2:
                            st.pyplot(plot_mfcc(audio_path))
                    
                    with tab2:
                        # Transcribe audio
                        transcription = transcribe_audio(whisper_model, audio_path)
                        st.subheader("Full Transcription")
                        st.write(transcription)
                    
                    with tab3:
                        st.subheader("Technical Processing Details")
                        st.write("""
                        **Processing Pipeline:**
                        1. Video downloaded from URL
                        2. Audio extracted and converted to WAV format
                        3. Audio preprocessed (resampled to 16kHz, converted to mono)
                        4. Spectral features extracted (MFCCs, spectrograms)
                        5. Accent classification using VoxLingua107 model
                        6. Speech transcription using Whisper
                        """)
                        
                        # Show file info
                        audio_stats = os.stat(audio_path)
                        st.write(f"**Audio File:** {audio_path}")
                        st.write(f"**File Size:** {audio_stats.st_size / (1024 * 1024):.2f} MB")
                        st.write(f"**Duration:** {librosa.get_duration(filename=audio_path):.2f} seconds")
                        
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()
