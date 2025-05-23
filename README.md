
# 🎙️ English Accent Detection Tool

This project is a **Streamlit-based web app** that detects the **English accent** (e.g., American, British, Indian) from a spoken video or audio input. It also provides **speech transcription** and a **waveform visualization** using machine learning models.

Powered by:

* [`OpenAI Whisper`](https://github.com/openai/whisper) for transcription
* [`SpeechBrain`](https://github.com/speechbrain/speechbrain) for accent detection
* [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) for audio extraction
* Streamlit for an interactive web UI

---

## 🔍 Features

* 🎯 Detects English accents (US, UK, Indian, Australian, etc.)
* 📝 Transcribes speech to text using Whisper
* 📊 Visualizes audio waveform
* 🎧 Accepts video/audio URLs (YouTube, Loom, MP4, etc.)
* 🧠 Uses pretrained ML models from Hugging Face and OpenAI

---

## 🧱 Tech Stack

| Component     | Technology                            |
| ------------- | ------------------------------------- |
| UI            | Streamlit                             |
| Transcription | OpenAI Whisper                        |
| Accent Model  | SpeechBrain ECAPA-TDNN (VoxLingua107) |
| Audio Tools   | yt-dlp, ffmpeg, pydub, librosa        |
| Visualization | Matplotlib, Librosa                   |

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/accent-detection-app.git
cd accent-detection-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Hugging Face token

Create a file at `.streamlit/secrets.toml` with the following content:

```toml
HUGGINGFACE_TOKEN = "your_huggingface_token_here"
```

You can get a token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Run the app

```bash
streamlit run app.py
```

---

## 📥 Input Formats

* Paste a **YouTube**, **Loom**, or **direct MP4/M4A/WAV** URL.
* The app will extract and analyze the audio.

---


## 🛠️ Troubleshooting

* Ensure `ffmpeg` is installed and available in your system path.
* Some private or region-locked videos may fail to download.
* Use direct audio URLs (MP3/WAV) if yt-dlp fails.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

* [OpenAI Whisper](https://github.com/openai/whisper)
* [SpeechBrain](https://github.com/speechbrain/speechbrain)
* [Hugging Face](https://huggingface.co/)
* [LibriVox](https://librivox.org/)
* [Streamlit](https://streamlit.io)



