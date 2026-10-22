# 🤟 Speech & Sign Converter

> An accessibility-focused application that converts spoken language into visual sign representations.

### 🌐 Live Website

👉 https://speech-sign-converter-x33umgk8nv8pvgbhyjutkj.streamlit.app/

Speech & Sign Converter uses speech recognition, multilingual language processing, emotion analysis, and a custom sign database to transform spoken communication into an easy-to-understand visual format.

---

## ✨ Features

* 🎙️ **Speech-to-Text** using OpenAI Whisper
* 📁 **Audio & Video File Support**
* 🌍 **Multilingual Speech Processing**
* 🔄 **Automatic Language Detection**
* 🤟 **Speech-to-Sign Conversion**
* 😊 **Voice Emotion Detection**
* 🔤 **Fingerspelling Fallback** for unsupported words
* 📊 **Sign Coverage Analysis**
* 🎧 **Audio Preprocessing & Normalization**
* 📚 **Custom Sign Database**
* ⚡ **Interactive Streamlit Interface**

### 🌐 Supported Languages

* 🇬🇧 English
* 🇮🇳 Hindi
* 🇮🇳 Gujarati
* 🇮🇳 Kannada
* 🇮🇳 Tamil
* 🇮🇳 Telugu

### 😊 Emotion Detection

The application analyzes acoustic and prosodic characteristics of speech to estimate emotions such as:

* 😊 Happy
* 😢 Sad
* 😠 Angry
* 😨 Fear
* 😮 Surprise
* 🤢 Disgust
* 😐 Neutral

---

## 💡 Overview

Communication can become challenging when speaking individuals and people who rely on sign language do not share the same communication method.

**Speech & Sign Converter** explores how AI-powered speech processing can help bridge this gap by converting spoken language into visual sign representations.

The application first converts speech into text, processes the recognized language, analyzes the emotional characteristics of the audio, and then maps the recognized words to corresponding signs from a custom database.

When a direct sign mapping is unavailable, the system uses a fingerspelling fallback so that the information is not completely lost.

---

## 🔄 How It Works

```text
🎙️ Speech / Audio / Video
          │
          ▼
   🎧 Audio Extraction
          │
          ▼
   🧹 Audio Preprocessing
          │
          ▼
     🧠 Whisper ASR
          │
          ▼
      📝 Transcript
          │
          ▼
   🌍 Language Processing
          │
          ├──────────────► 😊 Emotion Analysis
          │
          ▼
     🔤 Word Processing
          │
          ▼
   📚 Sign Database Lookup
          │
       ┌──┴──┐
       │     │
     Found  Not Found
       │     │
       ▼     ▼
    🤟 Sign  🔤 Fingerspelling
       │     │
       └──┬──┘
          ▼
     🤟 Sign Output
```

---

## 🧠 Core Components

### 🎙️ Speech Recognition

The application uses **OpenAI Whisper** to convert spoken audio into text.

Whisper enables the system to process speech across multiple supported languages.

### 🎧 Audio Processing

Uploaded audio is processed before transcription to improve consistency.

The preprocessing pipeline includes:

* Converting audio to mono
* Resampling audio to 16 kHz
* Trimming silence
* Normalizing audio levels
* Preparing audio for transcription and emotion analysis

### 🌍 Multilingual Processing

The application supports English and several Indian languages.

| Language | Code |
| -------- | ---- |
| English  | `en` |
| Hindi    | `hi` |
| Gujarati | `gu` |
| Kannada  | `kn` |
| Tamil    | `ta` |
| Telugu   | `te` |

Language translation is handled using **Deep Translator**.

### 😊 Emotion Analysis

The application analyzes acoustic and prosodic characteristics of speech to estimate the speaker's emotional state.

The analysis considers characteristics such as:

* Pitch
* Energy
* Amplitude
* Spectral features
* Speech characteristics

The detected emotion provides additional context alongside the generated sign representation.

### 🤟 Sign Conversion

Once speech has been converted into text, the application processes the recognized words individually.

Each word is searched in the custom sign database.

For example:

```text
Input:
hello friend

Output:
👋 🤝
```

The database contains mappings across categories such as:

* Greetings
* Emotions
* People
* Objects
* Places
* Actions
* Questions
* Numbers
* Colors
* Nature
* Activities
* Health
* General concepts

### 🔤 Fingerspelling Fallback

If a word does not have a direct mapping in the sign database, the application uses a fingerspelling representation instead of simply ignoring the word.

This allows unsupported words to remain represented in the output.

Example:

```text
Input:
hello astronaut

Output:
👋 + 🔤 fingerspelled representation
```

### 📊 Sign Coverage

The application calculates the percentage of recognized words that have a direct mapping in the sign database.

For example:

```text
hello  → 👋
friend → 🤝
happy  → 😊

Sign Coverage: 100%
```

If some words are unavailable:

```text
hello     → 👋
astronaut → 🔤 fingerspelled

Sign Coverage: 50%
```

---

## 🗂️ Project Structure

```text
speech-sign-converter/
│
├── app.py
├── sign_database.csv
├── requirements.txt
└── README.md
```

### `app.py`

Contains the main application logic, including:

* Streamlit interface
* Audio processing
* Speech recognition
* Language processing
* Emotion analysis
* Sign conversion
* Fingerspelling fallback

### `sign_database.csv`

Contains the word-to-sign mappings used by the application.

Example:

```csv
word,sign,category
hello,👋,greeting
happy,😊,feeling
friend,🤝,people
water,💧,object
```

### `requirements.txt`

Contains all Python dependencies required to run the application.

---

## 🛠️ Technology Stack

| Technology        | Purpose                |
| ----------------- | ---------------------- |
| Python            | Core application       |
| Streamlit         | Web interface          |
| OpenAI Whisper    | Speech recognition     |
| SpeechRecognition | Speech processing      |
| Deep Translator   | Language translation   |
| NumPy             | Numerical processing   |
| Pandas            | Data processing        |
| SoundFile         | Audio processing       |
| MoviePy           | Video audio extraction |
| CSV               | Sign database          |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/prachikaneriya23/speech-sign-converter.git
```

### 2. Navigate to the Project

```bash
cd speech-sign-converter
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

---

## 🎯 Usage

1. Open the application.
2. Upload an audio or supported video file.
3. Select a language or use automatic detection.
4. Allow the application to process the uploaded media.
5. View the generated speech transcription.
6. View the detected emotional characteristics.
7. View the corresponding visual sign representations.
8. Check the sign coverage of the processed text.
9. Unsupported words are handled through fingerspelling.

---

## 📁 Supported Media

The application can process audio files and extract audio from supported video formats.

### Video Formats

```text
.mp4
.mov
.avi
.webm
.mkv
```

---

## 🔮 Future Enhancements

Some possible improvements include:

* 🎥 Real-time speech-to-sign conversion
* 🎙️ Live microphone input
* 🤟 Animated sign-language avatars
* 🧠 Context-aware sign generation
* 📝 Sentence-level sign-language grammar
* 📱 Mobile application
* 🌐 Support for additional regional languages
* 📚 Larger and professionally validated sign datasets
* 🤖 AI-powered contextual sign prediction
* 🎞️ 3D or human sign-language animation

---

## ⚠️ Limitations

* The current system primarily relies on a word-to-sign database for visual representation.
* Direct word mapping does not always follow the grammar or structure of natural sign languages.
* Emotion detection depends on the quality and characteristics of the input audio.
* Words that are not available in the database are represented through fingerspelling.
* The current implementation is intended as an accessibility-focused prototype rather than a complete replacement for professional sign-language interpretation.

---

## 🌍 Purpose

Speech & Sign Converter explores the use of AI and speech technologies to create more accessible communication tools.

By combining speech recognition, multilingual processing, emotion analysis, and visual sign representation, the project provides a foundation for developing more advanced accessibility solutions.

---

## 🤝 Contributing

Contributions and improvements are welcome!

### Steps to Contribute

1. Fork the repository.
2. Create a new branch:

```bash
git checkout -b feature/your-feature
```

3. Make your changes.
4. Commit your changes:

```bash
git commit -m "Add your feature"
```

5. Push your branch:

```bash
git push origin feature/your-feature
```

6. Open a Pull Request.

---

## 👩‍💻 Author

### Prachi Kaneriya

GitHub: **[@prachikaneriya23](https://github.com/prachikaneriya23)**

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐.

---

## 📄 License

This project is intended for educational, research, and accessibility-focused purposes.

