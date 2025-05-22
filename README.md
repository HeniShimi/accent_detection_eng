# English Accent Analyzer

## Description

This tool analyzes a speaker's English accent from an online video. It is designed to help evaluate spoken English for hiring purposes by detecting the type of English accent (American, British, Australian, etc.) and providing a confidence score.

## Features

- Accepts a public video URL (YouTube, Loom, direct MP4, etc.)
- Extracts audio from the video
- Analyzes the speaker's accent to detect English language speaking candidates
- Provides accent classification (British, American, Australian, etc.)
- Calculates a confidence score (0-100%)
- Generates a detailed explanation of accent characteristics

## Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio extraction)
- An OpenAI API key (for transcription and accent analysis)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/accent-analyzer.git
cd accent-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

#### On Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### On macOS (with Homebrew)
```bash
brew install ffmpeg
```

#### On Windows
Download FFmpeg from [the official site](https://ffmpeg.org/download.html) and add it to your PATH.

### 5. Configure OpenAI API key

Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your-api-key
```

Or set it as an environment variable:

```bash
export OPENAI_API_KEY=your-api-key  # On Windows: set OPENAI_API_KEY=your-api-key
```

## Usage

### Run the Streamlit application

```bash
streamlit run src/app.py
```

The application will be accessible at http://localhost:8501 in your browser.

### User Interface

1. Enter a public video URL in the provided field
2. Click "Analyze Accent"
3. Wait for processing
4. View the analysis results

### Command-line usage

You can also test the tool from the command line:

```bash
python src/test_accent_analyzer.py --url "https://www.youtube.com/watch?v=example"
```

To run the complete test suite:

```bash
python src/test_accent_analyzer.py --suite
```

## Project Structure

```
accent_analyzer/
├── src/
│   ├── app.py                  # Streamlit application
│   ├── video_processor.py      # Video extraction and audio processing module
│   ├── accent_analyzer.py      # Accent analysis module
│   └── test_accent_analyzer.py # Test script
├── config.toml                 # Streamlit configuration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (to be created)
└── README.md                   # This file
```

## Deployment

### Deployment on Streamlit Cloud

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Deploy the application by selecting the `src/app.py` file
4. Add your OpenAI API key in the application secrets

### Deployment on Heroku

1. Create an account on [Heroku](https://heroku.com)
2. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
3. Log in to Heroku and create a new application:

```bash
heroku login
heroku create accent-analyzer-app
```

4. Add a buildpack for FFmpeg:

```bash
heroku buildpacks:add https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest.git
heroku buildpacks:add heroku/python
```

5. Configure the OpenAI API key:

```bash
heroku config:set OPENAI_API_KEY=your-api-key
```

6. Deploy the application:

```bash
git push heroku main
```

## Test Examples

Here are some example URLs you can use to test the application:

- **American accent**: https://www.youtube.com/watch?v=3FtGOHUkEzI
- **British accent**: https://www.youtube.com/watch?v=qYlmFISLO9M
- **Australian accent**: https://www.youtube.com/watch?v=4LvWYP7839Q
- **Indian accent**: https://www.youtube.com/watch?v=QYlVJlmjLEc

## Test Scenarios

The tool has been tested with the following scenarios:

1. **Video with clear American accent**
   - Expected result: "American English" classification with high confidence score (>85%)

2. **Video with British accent**
   - Expected result: "British English" classification with appropriate confidence score

3. **Video with non-native accent**
   - Expected result: Classification of the closest accent with moderate confidence score

4. **Video in a language other than English**
   - Expected result: Message indicating that the detected language is not English

5. **Video with poor audio quality**
   - Expected result: Analysis attempt with indication of limitation due to quality

## Limitations

- The tool requires an internet connection to download videos and use the OpenAI API
- Very long videos may take time to process
- Analysis quality depends on the audio quality of the video
- The tool is optimized for English accents only

## Troubleshooting

### Common issues

1. **Error "FFmpeg not found"**
   - Make sure FFmpeg is properly installed and accessible in your PATH

2. **Error "API key not found"**
   - Verify that you have correctly configured your OpenAI API key

3. **Error downloading video**
   - Check that the URL is valid and the video is publicly accessible
   - Make sure yt-dlp is up to date: `pip install -U yt-dlp`

4. **Slow performance**
   - Long videos may take time to process
   - Try with a shorter or better quality video

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or suggestions, please contact [your-email@example.com].
