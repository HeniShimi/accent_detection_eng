"""
Deployment script for the Streamlit application.
This script configures the environment and launches the application.
"""
import os
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_dependencies():
    """Check that system dependencies are installed."""
    try:
        # Check FFmpeg
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✅ FFmpeg is correctly installed.")
        
        # Check yt-dlp
        subprocess.run(['yt-dlp', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✅ yt-dlp is correctly installed.")
        
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"❌ Missing dependency: {e}")
        return False

def install_dependencies():
    """Install missing system dependencies."""
    try:
        # Install FFmpeg if needed
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print("FFmpeg is already installed.")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Installing FFmpeg...")
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
        
        # Install yt-dlp if needed
        try:
            subprocess.run(['yt-dlp', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print("yt-dlp is already installed.")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Installing yt-dlp...")
            subprocess.run(['pip', 'install', '-U', 'yt-dlp'], check=True)
        
        return True
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False

def check_api_key():
    """Check that the OpenAI API key is configured."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ No OpenAI API key found in environment variables.")
        print("Some features will be limited.")
        return False
    else:
        print("✅ OpenAI API key configured.")
        return True

def main():
    """Main function."""
    print("=== Configuring deployment environment ===")
    
    # Check dependencies
    if not check_dependencies():
        print("Installing missing dependencies...")
        if not install_dependencies():
            print("❌ Unable to install dependencies. Please install them manually.")
            sys.exit(1)
    
    # Check API key
    check_api_key()
    
    print("\n=== Starting Streamlit application ===")
    print("The application will be accessible at http://localhost:8501")
    
    # Launch Streamlit application
    try:
        subprocess.run(['streamlit', 'run', 'src/app.py'], check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
