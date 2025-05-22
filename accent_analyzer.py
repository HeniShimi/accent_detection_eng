"""
Module for English accent analysis from audio using Whisper for transcription
and Hugging Face model for accent classification.
"""
import os
import logging
import numpy as np
import librosa
import json
import tempfile
import whisper
import torch
from transformers import pipeline
from typing import Dict, Tuple, List, Optional

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccentAnalyzer:
    """Class for analyzing English accent in an audio file."""
    
    # Definition of main English accents
    ENGLISH_ACCENTS = [
        "American (US)",
        "British (UK)",
        "Australian",
        "Canadian",
        "Indian",
        "Irish",
        "Scottish",
        "South African",
        "New Zealand",
        "Non-native"
    ]
    
    def __init__(self, openai_api_key=None):
        """
        Initialize the accent analyzer.
        
        Args:
            openai_api_key (str, optional): Not used with local models, kept for compatibility.
        """
        # Load the Whisper model (smaller model for faster processing)
        logger.info("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
        
        # Load the Hugging Face accent classification model
        logger.info("Loading Hugging Face accent classification model...")
        self.accent_classifier = pipeline(
            "audio-classification", 
            model="Jzuluaga/accent-id-commonaccent_xlsr-en-english"
        )
        logger.info("Accent classification model loaded successfully")
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio to text using local Whisper model.
        
        Args:
            audio_path (str): Path to the audio file to transcribe.
            
        Returns:
            Dict: Transcription result containing text and metadata.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        
        try:
            logger.info(f"Transcribing audio with local Whisper: {audio_path}")
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_path)
            
            # Format result similar to OpenAI API response
            formatted_result = {
                "text": result["text"],
                "language": result.get("language", "en"),
                "segments": result.get("segments", [])
            }
            
            logger.info("Transcription successful")
            return formatted_result
        
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def extract_audio_features(self, audio_path: str) -> Dict:
        """
        Extract relevant audio features for accent analysis.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            Dict: Extracted audio features.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        
        try:
            logger.info(f"Extracting audio features: {audio_path}")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Extract pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[magnitudes > 0]) if np.any(magnitudes > 0) else 0
            
            # Extract rhythm
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Extract energy
            energy = np.mean(librosa.feature.rms(y=y))
            
            features = {
                "mfcc_mean": mfcc_mean.tolist(),
                "mfcc_std": mfcc_std.tolist(),
                "pitch_mean": float(pitch_mean),
                "tempo": float(tempo),
                "energy": float(energy)
            }
            
            logger.info("Audio feature extraction successful")
            return features
        
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            raise RuntimeError(f"Feature extraction failed: {e}")
    
    def analyze_accent_with_huggingface(self, audio_path: str) -> Dict:
        """
        Analyze accent using Hugging Face model.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            Dict: Accent analysis result.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        
        try:
            logger.info("Analyzing accent with Hugging Face model")
            
            # Run accent classification
            classification = self.accent_classifier(audio_path)
            
            # Get the top prediction
            top_prediction = classification[0]
            accent_label = top_prediction["label"]
            confidence = top_prediction["score"] * 100  # Convert to percentage
            
            # Map the model's label to our format
            accent_mapping = {
                "us": "American (US)",
                "england": "British (UK)",
                "australia": "Australian",
                "canada": "Canadian",
                "indian": "Indian",
                "african": "African",
                "scotland": "Scottish",
                "ireland": "Irish",
                "wales": "Welsh",
                "hongkong": "Hong Kong",
                "philippines": "Filipino",
                "malaysia": "Malaysian",
                "singapore": "Singaporean"
            }
            
            detected_accent = accent_mapping.get(accent_label, "Unknown")
            
            # Generate explanation based on the detected accent
            explanations = {
                "American (US)": "The speech contains typical American pronunciation patterns, characterized by rhotic 'r' sounds and specific vowel qualities.",
                "British (UK)": "The speech shows characteristic British intonation, non-rhotic 'r' sounds, and distinctive 't' pronunciation.",
                "Australian": "The speech has distinctive Australian vowel sounds, rising intonation, and characteristic expressions.",
                "Indian": "The speech demonstrates rhythmic patterns and consonant pronunciation common in Indian English.",
                "Canadian": "The speech contains subtle Canadian pronunciation features, including Canadian raising of diphthongs.",
                "Scottish": "The speech exhibits distinctive Scottish vowel sounds and strong 'r' pronunciation.",
                "Irish": "The speech shows melodic intonation patterns and vowel sounds characteristic of Irish English.",
                "Welsh": "The speech contains the distinctive musicality and consonant pronunciation of Welsh English.",
                "African": "The speech demonstrates rhythmic patterns and tonal qualities common in African varieties of English."
            }
            
            explanation = explanations.get(
                detected_accent, 
                f"The speech was classified as {detected_accent} accent based on pronunciation patterns and speech characteristics."
            )
            
            # Include the top 3 predictions for reference
            top_predictions = []
            for pred in classification[:3]:
                mapped_accent = accent_mapping.get(pred["label"], pred["label"])
                top_predictions.append({
                    "accent": mapped_accent,
                    "confidence": pred["score"] * 100
                })
            
            result = {
                "accent": detected_accent,
                "confidence": confidence,
                "explanation": explanation,
                "top_predictions": top_predictions
            }
            
            logger.info(f"Accent analysis successful: {result['accent']}")
            return result
        
        except Exception as e:
            logger.error(f"Error during accent analysis: {e}")
            raise RuntimeError(f"Accent analysis failed: {e}")
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Complete analysis of an audio file to detect English accent.
        
        Args:
            audio_path (str): Path to the audio file to analyze.
            
        Returns:
            Dict: Accent analysis result.
        """
        try:
            # Step 1: Transcribe audio
            transcription_result = self.transcribe_audio(audio_path)
            transcription_text = transcription_result.get("text", "")
            
            # Check if transcription is in English
            if not transcription_result.get("language") == "en":
                return {
                    "is_english": False,
                    "detected_language": transcription_result.get("language", "unknown"),
                    "message": "The detected language is not English."
                }
            
            # Step 2: Analyze accent with Hugging Face model
            accent_analysis = self.analyze_accent_with_huggingface(audio_path)
            
            # Step 3: Compile results
            result = {
                "is_english": True,
                "accent": accent_analysis.get("accent", "Unknown"),
                "confidence": accent_analysis.get("confidence", 0),
                "explanation": accent_analysis.get("explanation", ""),
                "transcription": transcription_text,
                "top_predictions": accent_analysis.get("top_predictions", [])
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error during audio analysis: {e}")
            return {
                "error": str(e),
                "message": "An error occurred during audio analysis."
            }


if __name__ == "__main__":
    # Simple test
    analyzer = AccentAnalyzer()
    test_audio = "path/to/test/audio.wav"  # Replace with a real path
    
    if os.path.exists(test_audio):
        try:
            result = analyzer.analyze_audio(test_audio)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Test audio file not found: {test_audio}")
