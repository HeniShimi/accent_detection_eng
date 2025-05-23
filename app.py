import streamlit as st
import whisper
import os
import numpy as np
import librosa
import tempfile
import torch
import soundfile as sf
import torchaudio
import yt_dlp
from pydub import AudioSegment
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier

# Initialize models
@st.cache_resource
def load_models():
    # Whisper for transcription
    whisper_model = whisper.load_model("base")
    
    # SpeechBrain for accent detection
    accent_model = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="tmp_model"
    )
    
    return whisper_model, accent_model

# Accent label mapping for VoxLingua107
ACCENT_LABELS = {
    "en": "General English",
    "en-us": "American English",
    "en-gb": "British English",
    "en-au": "Australian English",
    "en-in": "Indian English",
    "en-ca": "Canadian English",
    "en-ie": "Irish English",
    "en-nz": "New Zealand English",
    "en-za": "South African English"
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
        
        if not os.path.exists(audio_path):
            audio_path = os.path.join(temp_dir, "audio.m4a")
            if not os.path.exists(audio_path):
                raise FileNotFoundError("Audio extraction failed")
        
        if not audio_path.endswith('.wav'):
            sound = AudioSegment.from_file(audio_path)
            audio_path = os.path.join(temp_dir, "audio.wav")
            sound.export(audio_path, format="wav")
        
        return audio_path
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

def analyze_accent(audio_path, accent_model):
    try:
        # SpeechBrain expects 16kHz mono audio
        signal = accent_model.load_audio(audio_path)
        
        # Get prediction
        prediction = accent_model.classify_batch(signal)
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(prediction[2], 3)
        
        # Convert to readable results
        results = []
        for i in range(3):
            lang_code = prediction[3][top3_indices[i]]
            accent = ACCENT_LABELS.get(lang_code, lang_code)
            confidence = top3_probs[i].item() * 100
            results.append((accent, confidence))
        
        return results[0][0], results[0][1], results
    
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

def main():
    st.set_page_config(page_title="Accent Analyzer", layout="wide")
    st.title("🎙️ English Accent Detection Tool")
    
    video_url = st.text_input("Enter video URL (YouTube/Loom/MP4):")
    
    if st.button("Analyze Accent") and video_url:
        with st.spinner("Processing..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Load models
                    whisper_model, accent_model = load_models()
                    
                    # Download audio
                    audio_path = download_and_extract_audio(video_url, temp_dir)
                    if not audio_path:
                        return
                    
                    st.audio(audio_path, format="audio/wav")
                    
                    # Analyze
                    accent, confidence, all_results = analyze_accent(audio_path, accent_model)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Primary Result")
                        st.metric("Detected Accent", accent)
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    with col2:
                        st.subheader("Other Possibilities")
                        for alt_accent, alt_conf in all_results[1:]:
                            st.write(f"{alt_accent}: {alt_conf:.1f}%")
                            st.progress(int(alt_conf))
                    
                    # Transcription
                    st.subheader("Transcription")
                    st.write(transcribe_audio(whisper_model, audio_path))
                    
                    # Visualizations
                    st.subheader("Audio Analysis")
                    fig, ax = plt.subplots()
                    y, sr = librosa.load(audio_path)
                    librosa.display.waveshow(y, sr=sr, ax=ax)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()
