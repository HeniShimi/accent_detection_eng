import os
import tempfile
import subprocess
import logging
from urllib.parse import urlparse

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Class for handling video download and audio extraction."""
    
    def __init__(self, temp_dir=None):
        """
        Initialize the video processor.
        
        Args:
            temp_dir (str, optional): Temporary directory to store files.
                                     If None, uses the system's temporary directory.
        """
        self.temp_dir = temp_dir if temp_dir else tempfile.gettempdir()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check that external dependencies are installed."""
        try:
            # Check FFmpeg
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logger.info("FFmpeg is correctly installed.")
            
            # Check yt-dlp
            subprocess.run(['yt-dlp', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logger.info("yt-dlp is correctly installed.")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"Missing dependency: {e}")
            raise RuntimeError("Required dependencies (FFmpeg, yt-dlp) are not installed.")
    
    def validate_url(self, url):
        """
        Validate the video URL.
        
        Args:
            url (str): URL of the video to validate.
            
        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            logger.error(f"Invalid URL: {e}")
            return False
    
    def download_video(self, url):
        """
        Download a video from its URL.
        
        Args:
            url (str): URL of the video to download.
            
        Returns:
            str: Path to the downloaded video file.
        """
        if not self.validate_url(url):
            raise ValueError("Invalid video URL")
        
        # Create a unique temporary filename
        video_id = f"video_{os.urandom(4).hex()}"
        output_path = os.path.join(self.temp_dir, f"{video_id}.%(ext)s")
        
        try:
            # Use yt-dlp to download the video
            logger.info(f"Downloading video from {url}")
            cmd = [
                'yt-dlp',
                '--no-playlist',
                '--quiet',
                '-o', output_path,
                url
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Find the downloaded file (with the correct extension)
            for file in os.listdir(self.temp_dir):
                if file.startswith(video_id):
                    video_path = os.path.join(self.temp_dir, file)
                    logger.info(f"Video downloaded successfully: {video_path}")
                    return video_path
            
            raise FileNotFoundError("Video file not found after download")
        
        except subprocess.SubprocessError as e:
            logger.error(f"Error downloading video: {e}")
            raise RuntimeError(f"Failed to download video: {e}")
    
    def extract_audio(self, video_path, audio_format='wav', sample_rate=16000):
        """
        Extract audio from a video.
        
        Args:
            video_path (str): Path to the video file.
            audio_format (str, optional): Output audio format. Default 'wav'.
            sample_rate (int, optional): Sample rate in Hz. Default 16000.
            
        Returns:
            str: Path to the extracted audio file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        
        # Create a filename for the audio
        audio_id = f"audio_{os.urandom(4).hex()}"
        audio_path = os.path.join(self.temp_dir, f"{audio_id}.{audio_format}")
        
        try:
            # Use FFmpeg to extract audio
            logger.info(f"Extracting audio from {video_path}")
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-ar', str(sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite if file exists
                audio_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"Audio extracted successfully: {audio_path}")
            return audio_path
        
        except subprocess.SubprocessError as e:
            logger.error(f"Error extracting audio: {e}")
            raise RuntimeError(f"Failed to extract audio: {e}")
    
    def process_video_url(self, url):
        """
        Process a video URL: download the video and extract audio.
        
        Args:
            url (str): URL of the video to process.
            
        Returns:
            str: Path to the extracted audio file.
        """
        video_path = self.download_video(url)
        audio_path = self.extract_audio(video_path)
        
        # Remove the video to save space
        try:
            os.remove(video_path)
            logger.info(f"Temporary video file removed: {video_path}")
        except OSError as e:
            logger.warning(f"Unable to remove temporary video file: {e}")
        
        return audio_path
    
    def cleanup(self, file_path):
        """
        Clean up temporary files.
        
        Args:
            file_path (str): Path to the file to remove.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file removed: {file_path}")
        except OSError as e:
            logger.warning(f"Unable to remove temporary file: {e}")


if __name__ == "__main__":
    # Simple test
    processor = VideoProcessor()
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example URL
    try:
        audio_path = processor.process_video_url(test_url)
        print(f"Audio extracted successfully: {audio_path}")
        processor.cleanup(audio_path)
    except Exception as e:
        print(f"Error: {e}")
