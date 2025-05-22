"""
Module for English accent analysis from audio.
"""
import os
import logging
import numpy as np
import librosa
import openai
import json
import tempfile
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
            openai_api_key (str, optional): OpenAI API key for transcription and analysis.
                                           If None, uses the OPENAI_API_KEY environment variable.
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Some features will be limited.")
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio to text using OpenAI's Whisper.
        
        Args:
            audio_path (str): Path to the audio file to transcribe.
            
        Returns:
            Dict: Transcription result containing text and metadata.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for transcription")
        
        try:
            openai.api_key = self.openai_api_key
            
            logger.info(f"Transcribing audio: {audio_path}")
            with open(audio_path, "rb") as audio_file:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    language="en"
                )
            
            logger.info("Transcription successful")
            return response
        
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
    
    def analyze_accent_with_ai(self, transcription: str) -> Dict:
        """
        Analyze accent using OpenAI API.
        
        Args:
            transcription (str): Transcribed text to analyze.
            
        Returns:
            Dict: Accent analysis result.
        """
        if not transcription:
            raise ValueError("Empty transcription")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for accent analysis")
        
        try:
            openai.api_key = self.openai_api_key
            
            logger.info("Analyzing accent with AI")
            
            # Build prompt for accent analysis
            prompt = f"""
            Analyze the following English text transcription and determine the speaker's accent.
            
            Transcription: "{transcription}"
            
            Please provide:
            1. The most likely English accent (American, British, Australian, Canadian, Indian, Irish, Scottish, South African, New Zealand, or Non-native)
            2. A confidence score from 0 to 100
            3. A brief explanation of the accent characteristics detected
            
            Format your response as a JSON object with the following structure:
            {{
                "accent": "accent_name",
                "confidence": score,
                "explanation": "detailed explanation"
            }}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in English accent analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Extract and parse JSON response
            content = response.choices[0].message.content.strip()
            # Find the start and end of JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                logger.info(f"Accent analysis successful: {result['accent']}")
                return result
            else:
                # Fallback if JSON format is not detected
                logger.warning("JSON format not detected in response, creating manual structure")
                return {
                    "accent": "Unknown",
                    "confidence": 0,
                    "explanation": "Unable to analyze accent from transcription."
                }
        
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
            
            # Step 2: Extract audio features (for future reference)
            audio_features = self.extract_audio_features(audio_path)
            
            # Step 3: Analyze accent with AI
            accent_analysis = self.analyze_accent_with_ai(transcription_text)
            
            # Step 4: Compile results
            result = {
                "is_english": True,
                "accent": accent_analysis.get("accent", "Unknown"),
                "confidence": accent_analysis.get("confidence", 0),
                "explanation": accent_analysis.get("explanation", ""),
                "transcription": transcription_text
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error during audio analysis: {e}")
            return {
                "error": str(e),
                "message": "An error occurred during audio analysis."
            }


if __name__ == "__main__":
    # Simple test (requires OpenAI API key)
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
