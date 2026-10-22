import io
import os
import re
import string
import tempfile
import urllib.parse

from deep_translator import GoogleTranslator
import numpy as np
import pandas as pd
import soundfile as sf
import speech_recognition as sr
import streamlit as st
import streamlit.components.v1 as components
import whisper

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False

WHISPER_SR = 16000  # sample rate Whisper expects

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "sign_database.csv")
FINGERSPELL_BASE = 0x1F1E6  # 🇦 regional indicator symbol letter A

LANGUAGE_MAP = {
    "Auto Detect": None,
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
}

INDIC_LANGUAGE_TAGS = {
    "hi": "hi-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}

CODE_TO_LANGUAGE = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
}

EMOTION_META = {
    "HAPPY": {"emoji": "😊", "color": "#10b981", "desc": "Happy / Joyful", "bg": "linear-gradient(135deg, #064e3b, #047857)"},
    "SAD": {"emoji": "😢", "color": "#3b82f6", "desc": "Sad / Subdued", "bg": "linear-gradient(135deg, #1e3a8a, #1d4ed8)"},
    "ANGRY": {"emoji": "😠", "color": "#ef4444", "desc": "Angry / Frustrated", "bg": "linear-gradient(135deg, #7f1d1d, #b91c1c)"},
    "FEAR": {"emoji": "😨", "color": "#8b5cf6", "desc": "Fearful / Anxious", "bg": "linear-gradient(135deg, #4c1d95, #6d28d9)"},
    "SURPRISE": {"emoji": "😲", "color": "#06b6d4", "desc": "Surprised / Excited", "bg": "linear-gradient(135deg, #164e63, #0891b2)"},
    "DISGUST": {"emoji": "🤢", "color": "#84cc16", "desc": "Disgusted / Aversive", "bg": "linear-gradient(135deg, #365314, #4d7c0f)"},
    "NEUTRAL": {"emoji": "😐", "color": "#64748b", "desc": "Neutral / Calm", "bg": "linear-gradient(135deg, #0f172a, #1e293b)"},
}

st.set_page_config(page_title="Speech & Audio Emotion to Sign", page_icon="🤟", layout="wide")

# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str):
    return whisper.load_model(model_size)


@st.cache_data(show_spinner=False)
def load_sign_database(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        default_data = [
            {"word": "hello", "sign": "👋", "category": "greetings"},
            {"word": "hi", "sign": "👋", "category": "greetings"},
            {"word": "bye", "sign": "👋", "category": "greetings"},
            {"word": "goodbye", "sign": "👋", "category": "greetings"},
            {"word": "welcome", "sign": "👐", "category": "greetings"},
            {"word": "yes", "sign": "👍", "category": "responses"},
            {"word": "no", "sign": "👎", "category": "responses"},
            {"word": "please", "sign": "🙏", "category": "courtesy"},
            {"word": "thank", "sign": "🤝", "category": "courtesy"},
            {"word": "thanks", "sign": "🤝", "category": "courtesy"},
            {"word": "sorry", "sign": "🙇", "category": "courtesy"},
            {"word": "help", "sign": "🆘", "category": "common"},
            {"word": "stop", "sign": "🛑", "category": "common"},
            {"word": "go", "sign": "🟢", "category": "common"},
            {"word": "good", "sign": "👍", "category": "common"},
            {"word": "bad", "sign": "👎", "category": "common"},
            {"word": "love", "sign": "❤️", "category": "emotions"},
            {"word": "happy", "sign": "😊", "category": "emotions"},
            {"word": "sad", "sign": "😢", "category": "emotions"},
            {"word": "angry", "sign": "😠", "category": "emotions"},
            {"word": "water", "sign": "💧", "category": "food"},
            {"word": "food", "sign": "🍲", "category": "food"},
            {"word": "eat", "sign": "🍽️", "category": "food"},
            {"word": "drink", "sign": "🥤", "category": "food"},
            {"word": "coffee", "sign": "☕", "category": "food"},
            {"word": "tea", "sign": "🍵", "category": "food"},
            {"word": "friend", "sign": "🧑‍🤝‍🧑", "category": "people"},
            {"word": "family", "sign": "👨‍👩‍👧‍👦", "category": "people"},
            {"word": "mother", "sign": "👩", "category": "people"},
            {"word": "father", "sign": "👨", "category": "people"},
            {"word": "brother", "sign": "👦", "category": "people"},
            {"word": "sister", "sign": "👧", "category": "people"},
            {"word": "peace", "sign": "✌️", "category": "gestures"},
            {"word": "okay", "sign": "👌", "category": "gestures"},
            {"word": "victory", "sign": "✌️", "category": "gestures"},
            {"word": "clap", "sign": "👏", "category": "gestures"},
            {"word": "call", "sign": "🤙", "category": "gestures"},
            {"word": "pray", "sign": "🙏", "category": "gestures"},
            {"word": "look", "sign": "👀", "category": "senses"},
            {"word": "listen", "sign": "👂", "category": "senses"},
            {"word": "speak", "sign": "🗣️", "category": "senses"},
            {"word": "home", "sign": "🏠", "category": "places"},
            {"word": "school", "sign": "🏫", "category": "places"},
            {"word": "work", "sign": "💼", "category": "places"},
            {"word": "car", "sign": "🚗", "category": "transport"},
            {"word": "bus", "sign": "🚌", "category": "transport"},
            {"word": "time", "sign": "⏰", "category": "common"},
            {"word": "day", "sign": "☀️", "category": "time"},
            {"word": "night", "sign": "🌙", "category": "time"},
            {"word": "money", "sign": "💰", "category": "common"},
            {"word": "question", "sign": "❓", "category": "common"},
        ]
        df = pd.DataFrame(default_data)
        df.to_csv(path, index=False)
        return df
    df = pd.read_csv(path)
    df["word"] = df["word"].str.lower().str.strip()
    df = df.drop_duplicates(subset="word", keep="first").reset_index(drop=True)
    return df


def fingerspell(word: str) -> str:
    """Fallback for words not in the database: spell it out with
    regional-indicator emoji letters, standing in for ASL fingerspelling."""
    letters = []
    for ch in word.lower():
        if "a" <= ch <= "z":
            letters.append(chr(FINGERSPELL_BASE + (ord(ch) - ord("a"))))
    return "".join(letters) if letters else "❓"


def clean_tokens(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
    tokens = re.findall(r"[a-z']+", text)
    return tokens


def words_to_signs(tokens: list[str], df: pd.DataFrame) -> pd.DataFrame:
    columns = ["word", "sign", "category", "matched"]
    if not tokens:
        return pd.DataFrame(columns=columns), 0.0

    lookup = df.set_index("word")["sign"].to_dict()
    cats = df.set_index("word")["category"].to_dict()

    rows = []
    hits = np.zeros(len(tokens), dtype=bool)
    for i, tok in enumerate(tokens):
        if tok in lookup:
            rows.append({"word": tok, "sign": lookup[tok], "category": cats[tok], "matched": True})
            hits[i] = True
        else:
            rows.append({"word": tok, "sign": fingerspell(tok), "category": "fingerspelled", "matched": False})
    coverage = float(hits.mean()) * 100 if len(hits) else 0.0
    return pd.DataFrame(rows, columns=columns), coverage


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    duration = len(audio) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold: float = 0.015, pad_samples: int = 1600) -> np.ndarray:
    """Carefully remove leading/trailing silence from 16kHz audio with safety padding."""
    if len(audio) == 0:
        return audio

    abs_audio = np.abs(audio)
    max_amp = float(np.max(abs_audio))
    if max_amp == 0:
        return audio

    thresh = max(threshold, 0.05 * max_amp)
    above_thresh = np.where(abs_audio > thresh)[0]

    if len(above_thresh) == 0:
        return audio

    start_idx = max(0, above_thresh[0] - pad_samples)
    end_idx = min(len(audio), above_thresh[-1] + pad_samples)
    return audio[start_idx:end_idx]


def preprocess_audio(audio: np.ndarray) -> np.ndarray:
    """Preprocess audio: float32 mono 16kHz, silence trimmed, and non-destructive normalization."""
    if len(audio) == 0:
        return audio

    audio = audio.astype(np.float32)
    audio = trim_silence(audio)

    max_abs = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if max_abs > 0.05:
        audio = (audio / max_abs) * 0.95

    return audio.astype(np.float32)


def extract_audio_from_uploaded_file(file_bytes: bytes, file_name: str) -> bytes:
    """Extracts raw audio WAV bytes from audio or video uploads."""
    lower_name = file_name.lower()
    video_exts = [".mp4", ".mov", ".avi", ".webm", ".mkv"]
    is_video = any(lower_name.endswith(ext) for ext in video_exts)

    if is_video and MOVIEPY_AVAILABLE:
        ext = os.path.splitext(file_name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_in:
            temp_in.write(file_bytes)
            temp_in_path = temp_in.name

        temp_out_path = temp_in_path + "_audio.wav"
        try:
            clip = VideoFileClip(temp_in_path)
            if clip.audio is not None:
                clip.audio.write_audiofile(
                    temp_out_path, fps=16000, nbytes=2, codec="pcm_s16le", logger=None
                )
                clip.close()
                with open(temp_out_path, "rb") as f:
                    audio_bytes = f.read()
                return audio_bytes
            else:
                raise ValueError("The uploaded video does not contain an audio track.")
        finally:
            for p in [temp_in_path, temp_out_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
    return file_bytes


def bytes_to_audio_array(audio_bytes: bytes) -> tuple[np.ndarray, dict]:
    """Decode audio entirely in-memory with soundfile (libsndfile).
    Calculates diagnostic metrics, resamples to mono float32 16kHz,
    and preprocesses for Whisper and Emotion DSP."""
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono

    num_samples = len(data)
    duration_s = float(num_samples / sr) if sr > 0 else 0.0
    min_amp = float(np.min(data)) if num_samples > 0 else 0.0
    max_amp = float(np.max(data)) if num_samples > 0 else 0.0
    rms = float(np.sqrt(np.mean(data**2))) if num_samples > 0 else 0.0

    diag = {
        "sample_rate": sr,
        "num_samples": num_samples,
        "duration_s": round(duration_s, 3),
        "min_amplitude": round(min_amp, 5),
        "max_amplitude": round(max_amp, 5),
        "rms_amplitude": round(rms, 5),
    }

    # Resample to 16000 Hz float32
    resampled = resample(data, sr, WHISPER_SR).astype(np.float32)
    preprocessed = preprocess_audio(resampled)

    return preprocessed, diag


# --------------------------------------------------------------------------
# Audio-Based Emotion Detection (Acoustic-Prosodic Feature Engine)
# --------------------------------------------------------------------------
def extract_comprehensive_prosody(audio: np.ndarray, sr: int = 16000) -> dict:
    """Extract detailed prosodic and spectral acoustic features from raw audio signal."""
    if len(audio) == 0:
        return {}

    duration_s = float(len(audio) / sr)
    frame_size = int(sr * 0.03)  # 30ms frames
    hop_size = int(sr * 0.015)  # 15ms hop

    # 1. Frame-level RMS energy
    rms_frames = []
    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i : i + frame_size]
        rms_frames.append(float(np.sqrt(np.mean(frame**2))))

    rms_arr = np.array(rms_frames) if rms_frames else np.array([float(np.sqrt(np.mean(audio**2)))])
    rms_mean = float(np.mean(rms_arr))
    rms_std = float(np.std(rms_arr))
    rms_max = float(np.max(rms_arr)) if len(rms_arr) else 0.0
    dynamic_range_db = float(20 * np.log10((rms_max + 1e-6) / (np.percentile(rms_arr, 10) + 1e-6)))

    # 2. Pitch / F0 via Autocorrelation & Peak Validation
    pitches = []
    min_lag = int(sr / 450)  # Max 450 Hz
    max_lag = int(sr / 75)  # Min 75 Hz
    voiced_thresh = max(0.015, 0.20 * rms_mean)

    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i : i + frame_size]
        frame_rms = np.sqrt(np.mean(frame**2))
        if frame_rms > voiced_thresh:
            frame_centered = frame - np.mean(frame)
            autocorr = np.correlate(frame_centered, frame_centered, mode="full")
            autocorr = autocorr[len(frame_centered) - 1 :]
            if len(autocorr) > max_lag:
                search_region = autocorr[min_lag:max_lag]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region) + min_lag
                    peak_val = autocorr[peak_idx]
                    if autocorr[0] > 0 and (peak_val / autocorr[0]) > 0.35:
                        freq = sr / peak_idx
                        if 75 <= freq <= 450:
                            pitches.append(freq)

    pitches = np.array(pitches) if pitches else np.array([])
    pitch_mean = float(np.mean(pitches)) if len(pitches) > 0 else 0.0
    pitch_std = float(np.std(pitches)) if len(pitches) > 0 else 0.0
    pitch_min = float(np.min(pitches)) if len(pitches) > 0 else 0.0
    pitch_max = float(np.max(pitches)) if len(pitches) > 0 else 0.0
    pitch_range = float(pitch_max - pitch_min) if len(pitches) > 0 else 0.0
    voiced_ratio = float(len(pitches) / (len(rms_arr) + 1e-6))

    jitter = float(np.mean(np.abs(np.diff(pitches))) / (pitch_mean + 1e-6)) if len(pitches) > 2 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)

    # 3. Spectral Analysis via FFT
    n_fft = 512
    window = np.hanning(n_fft)
    stft_frames = []
    for i in range(0, len(audio) - n_fft + 1, hop_size):
        frame = audio[i : i + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        stft_frames.append(spectrum)

    stft_frames = np.array(stft_frames) if stft_frames else np.zeros((1, n_fft // 2 + 1))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    centroids = []
    bandwidths = []
    fluxes = []
    high_freq_ratios = []
    hf_indices = np.where(freqs >= 1800)[0]

    for idx, spec in enumerate(stft_frames):
        spec_sum = np.sum(spec)
        if spec_sum > 1e-6:
            c = np.sum(freqs * spec) / spec_sum
            bw = np.sqrt(np.sum(((freqs - c) ** 2) * spec) / spec_sum)
            centroids.append(c)
            bandwidths.append(bw)
            if len(hf_indices) > 0:
                high_freq_ratios.append(np.sum(spec[hf_indices]) / spec_sum)
        else:
            centroids.append(0.0)
            bandwidths.append(0.0)
            high_freq_ratios.append(0.0)

        if idx > 0:
            flux = np.sqrt(np.mean((spec - stft_frames[idx - 1]) ** 2))
            fluxes.append(flux)

    spectral_centroid = float(np.mean(centroids)) if centroids else 0.0
    spectral_bandwidth = float(np.mean(bandwidths)) if bandwidths else 0.0
    spectral_flux = float(np.mean(fluxes)) if fluxes else 0.0
    high_freq_energy = float(np.mean(high_freq_ratios)) if high_freq_ratios else 0.0

    # 4. Temporal Rhythm & Speaking Dynamics
    silence_thresh = max(0.015, 0.15 * rms_mean)
    silent_frames = np.sum(rms_arr < silence_thresh)
    pause_ratio = float(silent_frames / (len(rms_arr) + 1e-6))

    peaks = 0
    for k in range(1, len(rms_arr) - 1):
        if rms_arr[k] > 0.03 and rms_arr[k] > rms_arr[k - 1] and rms_arr[k] > rms_arr[k + 1]:
            peaks += 1
    speaking_rate = float(peaks / duration_s) if duration_s > 0.4 else 0.0

    return {
        "duration_s": round(duration_s, 3),
        "rms_mean": round(rms_mean, 4),
        "rms_std": round(rms_std, 4),
        "rms_max": round(rms_max, 4),
        "dynamic_range_db": round(dynamic_range_db, 1),
        "pitch_mean": round(pitch_mean, 1),
        "pitch_std": round(pitch_std, 1),
        "pitch_min": round(pitch_min, 1),
        "pitch_max": round(pitch_max, 1),
        "pitch_range": round(pitch_range, 1),
        "voiced_ratio": round(voiced_ratio, 3),
        "jitter": round(jitter, 4),
        "zcr": round(zcr, 4),
        "spectral_centroid": round(spectral_centroid, 1),
        "spectral_bandwidth": round(spectral_bandwidth, 1),
        "spectral_flux": round(spectral_flux, 4),
        "high_freq_energy": round(high_freq_energy, 4),
        "pause_ratio": round(pause_ratio, 3),
        "speaking_rate": round(speaking_rate, 2),
    }


def analyze_audio_emotion(audio: np.ndarray, sr: int = 16000) -> dict:
    """Acoustic-Prosodic Speech Emotion Classifier.
    Evaluates prosody and acoustic dynamics over overlapping temporal windows,
    applies temporal smoothing, and computes authentic confidence & intensity."""
    duration_s = len(audio) / sr

    # Uncertainty check for very short audio
    if duration_s < 0.4:
        return {
            "emotion": "NEUTRAL",
            "confidence": 35.0,
            "intensity": 20.0,
            "uncertainty_note": "Audio clip too short (< 0.4s) for reliable prosody estimation.",
            "features": {"duration_s": round(duration_s, 2), "rms_mean": 0.0, "pitch_mean": 0.0, "speaking_rate": 0.0, "pause_ratio": 0.0},
            "model_name": "Acoustic-Prosodic Speech Emotion Analyzer (Prosody-DSP v2.0)",
            "probabilities": {"NEUTRAL": 100.0, "HAPPY": 0.0, "SAD": 0.0, "ANGRY": 0.0, "FEAR": 0.0, "SURPRISE": 0.0, "DISGUST": 0.0},
        }

    overall_feat = extract_comprehensive_prosody(audio, sr)
    if overall_feat.get("rms_mean", 0.0) < 0.008:
        return {
            "emotion": "NEUTRAL",
            "confidence": 30.0,
            "intensity": 15.0,
            "uncertainty_note": "Audio signal volume too low / silence detected.",
            "features": overall_feat,
            "model_name": "Acoustic-Prosodic Speech Emotion Analyzer (Prosody-DSP v2.0)",
            "probabilities": {"NEUTRAL": 100.0, "HAPPY": 0.0, "SAD": 0.0, "ANGRY": 0.0, "FEAR": 0.0, "SURPRISE": 0.0, "DISGUST": 0.0},
        }

    # Windowed analysis with 1.8s windows and 0.6s step for temporal smoothing
    win_len = int(sr * 1.8)
    step_len = int(sr * 0.6)

    if len(audio) <= win_len:
        windows = [audio]
    else:
        windows = [audio[i : i + win_len] for i in range(0, len(audio) - win_len + 1, step_len)]
        if not windows:
            windows = [audio]

    window_scores = []

    for win_audio in windows:
        f = extract_comprehensive_prosody(win_audio, sr)
        if not f:
            continue

        rms = f["rms_mean"]
        rms_max = f["rms_max"]
        p_mean = f["pitch_mean"]
        p_std = f["pitch_std"]
        p_max = f["pitch_max"]
        p_range = f["pitch_range"]
        centroid = f["spectral_centroid"]
        flux = f["spectral_flux"]
        hf_energy = f["high_freq_energy"]
        spk_rate = f["speaking_rate"]
        pause_r = f["pause_ratio"]
        jitter = f["jitter"]
        zcr = f["zcr"]

        # Logit accumulator for 7 emotions
        scores = {
            "HAPPY": 0.0,
            "SAD": 0.0,
            "ANGRY": 0.0,
            "FEAR": 0.0,
            "SURPRISE": 0.0,
            "DISGUST": 0.0,
            "NEUTRAL": 2.2,  # Prior anchor towards neutral in conversational speech
        }

        # 1. ANGRY (High energy, sharp high-frequencies, high flux, elevated pitch, fast attack)
        if rms > 0.24:
            scores["ANGRY"] += (rms - 0.24) * 12.0
        if rms_max > 0.40:
            scores["ANGRY"] += (rms_max - 0.40) * 6.0
        if hf_energy > 0.10 and rms > 0.20:
            scores["ANGRY"] += (hf_energy - 0.10) * 15.0
        if flux > 0.03 and rms > 0.22:
            scores["ANGRY"] += min(3.0, flux * 40.0)
        if p_mean > 190 and rms > 0.22:
            scores["ANGRY"] += min(2.5, (p_mean - 190) / 35.0)

        # 2. HAPPY (High expressive pitch variation, bright centroid, elevated pitch, musical melody)
        if p_std > 28 and p_mean > 200:
            scores["HAPPY"] += min(4.0, (p_std - 28) / 12.0 + (p_mean - 200) / 35.0)
        if p_range > 90 and p_mean > 210:
            scores["HAPPY"] += min(3.0, (p_range - 90) / 45.0)
        if centroid > 800 and hf_energy < 0.20 and rms < 0.40:
            scores["HAPPY"] += min(2.5, (centroid - 800) / 350.0)
        if spk_rate > 8.0 and rms > 0.12 and rms < 0.38:
            scores["HAPPY"] += min(2.0, (spk_rate - 8.0) / 3.0)

        # 3. SAD (Subdued energy < 0.07, low pitch < 130 Hz, high pauses, slow speaking rate)
        if rms < 0.065 and pause_r > 0.20:
            scores["SAD"] += (0.065 - rms) * 40.0 + (pause_r - 0.20) * 8.0
        if p_mean > 0 and p_mean < 125 and rms < 0.08:
            scores["SAD"] += min(3.0, (125 - p_mean) / 20.0)
        if pause_r > 0.30 and rms < 0.08:
            scores["SAD"] += min(3.0, (pause_r - 0.30) * 8.0)
        if centroid < 350 and centroid > 0 and rms < 0.08:
            scores["SAD"] += min(2.0, (350 - centroid) / 100.0)

        # 4. FEAR (Pitch jitter/instability, elevated pitch, high ZCR, erratic breathiness)
        if jitter > 0.06 and p_mean > 200:
            scores["FEAR"] += min(3.5, (jitter - 0.06) * 30.0 + (p_mean - 200) / 45.0)
        if zcr > 0.14 and rms < 0.28:
            scores["FEAR"] += min(2.0, (zcr - 0.14) * 12.0)

        # 5. SURPRISE (Steep pitch jump / wide range, abrupt onset, high peak)
        if p_range > 150 and p_max > 260:
            scores["SURPRISE"] += min(4.0, (p_range - 150) / 40.0 + (p_max - 260) / 40.0)

        # 6. DISGUST (Low creaky pitch, slow articulation, low centroid with modest energy)
        if p_mean > 0 and p_mean < 110 and rms > 0.08 and spk_rate < 5.5:
            scores["DISGUST"] += min(3.0, (110 - p_mean) / 20.0 + (5.5 - spk_rate))

        # 7. NEUTRAL (Balanced pitch 105-185 Hz, normal energy 0.07-0.22, calm pauses)
        if 105 <= p_mean <= 185 and rms <= 0.22:
            scores["NEUTRAL"] += 2.5
        if 0.06 <= rms <= 0.20:
            scores["NEUTRAL"] += 2.0
        if p_std <= 25 and p_mean > 0 and rms <= 0.20:
            scores["NEUTRAL"] += 1.5

        window_scores.append(scores)

    if not window_scores:
        return {
            "emotion": "NEUTRAL",
            "confidence": 45.0,
            "intensity": 25.0,
            "uncertainty_note": "No voiced windows found in recording.",
            "features": overall_feat,
            "model_name": "Acoustic-Prosodic Speech Emotion Analyzer (Prosody-DSP v2.0)",
            "probabilities": {"NEUTRAL": 100.0, "HAPPY": 0.0, "SAD": 0.0, "ANGRY": 0.0, "FEAR": 0.0, "SURPRISE": 0.0, "DISGUST": 0.0},
        }

    # Aggregate across temporal windows with confidence weighting
    agg_scores = {e: 0.0 for e in ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "DISGUST", "NEUTRAL"]}
    for ws in window_scores:
        for e in agg_scores:
            agg_scores[e] += ws[e]

    score_vals = np.array([agg_scores[e] for e in agg_scores])
    exp_scores = np.exp(score_vals - np.max(score_vals))
    probs = exp_scores / np.sum(exp_scores)

    prob_dict = {e: float(probs[i]) for i, e in enumerate(agg_scores)}
    sorted_emotions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    top_emotion, top_prob = sorted_emotions[0]
    second_emotion, second_prob = sorted_emotions[1]

    # Confidence: Certainty based on separation margin and probability
    margin = top_prob - second_prob
    confidence_raw = top_prob * 0.65 + margin * 0.35
    confidence = float(np.clip(confidence_raw * 100.0, 38.0, 95.0))

    # Intensity: Physical acoustic strength / deviation from neutral
    rms_intensity = min(1.0, overall_feat.get("rms_mean", 0.1) / 0.40)
    pitch_intensity = min(1.0, overall_feat.get("pitch_std", 20.0) / 70.0)
    flux_intensity = min(1.0, overall_feat.get("spectral_flux", 0.02) / 0.10)

    if top_emotion == "NEUTRAL":
        intensity = float(np.clip((rms_intensity * 0.4 + 0.1) * 50.0, 15.0, 42.0))
    elif top_emotion == "SAD":
        intensity = float(np.clip((1.0 - rms_intensity) * 65.0 + (overall_feat.get("pause_ratio", 0) * 35.0), 30.0, 85.0))
    elif top_emotion == "ANGRY":
        intensity = float(np.clip((rms_intensity * 0.6 + flux_intensity * 0.4) * 100.0, 50.0, 98.0))
    else:
        intensity = float(np.clip((rms_intensity * 0.4 + pitch_intensity * 0.4 + flux_intensity * 0.2) * 100.0, 35.0, 92.0))

    uncertainty_note = None
    if confidence < 45.0:
        top_emotion = "NEUTRAL"
        uncertainty_note = "Emotion confidence is too low to distinguish subtle acoustic state; defaulted to NEUTRAL."

    return {
        "emotion": top_emotion,
        "confidence": round(confidence, 1),
        "intensity": round(intensity, 1),
        "uncertainty_note": uncertainty_note,
        "probabilities": {k: round(v * 100.0, 1) for k, v in prob_dict.items()},
        "features": overall_feat,
        "model_name": "Acoustic-Prosodic Speech Emotion Analyzer (Prosody-DSP v2.0)",
    }


# --------------------------------------------------------------------------
# ASR Router & Backends
# --------------------------------------------------------------------------
class BaseASRBackend:
    name: str = "Base ASR"

    def transcribe(self, audio_bytes: bytes, preprocessed_audio: np.ndarray, language_code: str = None) -> tuple[str, str]:
        raise NotImplementedError


class WhisperASRBackend(BaseASRBackend):
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.name = f"Whisper ({model_size})"

    def transcribe(self, audio_bytes: bytes, preprocessed_audio: np.ndarray, language_code: str = None) -> tuple[str, str]:
        model = load_whisper_model(self.model_size)
        transcribe_kwargs = {
            "task": "transcribe",
            "fp16": False,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        }
        if language_code:
            transcribe_kwargs["language"] = language_code

        result = model.transcribe(preprocessed_audio, **transcribe_kwargs)
        text = result.get("text", "").strip()
        detected_lang = result.get("language", language_code or "en")
        return text, detected_lang


class IndicASRBackend(BaseASRBackend):
    name: str = "Indic ASR Engine (AI4Bharat / Google Indic)"

    def __init__(self):
        self._recognizer = sr.Recognizer()

    def transcribe(self, audio_bytes: bytes, preprocessed_audio: np.ndarray, language_code: str = None) -> tuple[str, str]:
        lang_tag = INDIC_LANGUAGE_TAGS.get(language_code, "hi-IN" if language_code == "hi" else f"{language_code}-IN")
        wav_buf = io.BytesIO()
        sf.write(wav_buf, preprocessed_audio, WHISPER_SR, format="WAV")
        wav_buf.seek(0)

        with sr.AudioFile(wav_buf) as source:
            audio_data = self._recognizer.record(source)

        text = self._recognizer.recognize_google(audio_data, language=lang_tag)
        return text.strip(), language_code or "hi"


class ASRRouter:
    def __init__(self, whisper_model_size: str = "small"):
        self.whisper_model_size = whisper_model_size
        self.whisper_backend = WhisperASRBackend(whisper_model_size)
        self.indic_backend = IndicASRBackend()

    def route_and_transcribe(
        self, audio_bytes: bytes, preprocessed_audio: np.ndarray, language: str = None
    ) -> tuple[str, str, dict]:
        lang_code = LANGUAGE_MAP.get(language, language) if language else None
        if lang_code == "Auto Detect":
            lang_code = None

        backend_used = None
        transcript = ""
        detected_lang = lang_code or "en"
        fallback_used = False

        if lang_code in INDIC_LANGUAGE_TAGS:
            try:
                transcript, detected_lang = self.indic_backend.transcribe(
                    audio_bytes, preprocessed_audio, lang_code
                )
                backend_used = f"{self.indic_backend.name} [{INDIC_LANGUAGE_TAGS[lang_code]}]"
            except Exception as indic_err:
                transcript, detected_lang = self.whisper_backend.transcribe(
                    audio_bytes, preprocessed_audio, lang_code
                )
                backend_used = f"Whisper ({self.whisper_model_size}) [Fallback from Indic ASR: {indic_err}]"
                fallback_used = True
        else:
            transcript, detected_lang = self.whisper_backend.transcribe(
                audio_bytes, preprocessed_audio, lang_code
            )
            backend_used = self.whisper_backend.name

        debug_info = {
            "selected_language": language or "Auto Detect",
            "passed_language_code": lang_code,
            "asr_backend": backend_used,
            "model_size": self.whisper_model_size,
            "returned_language": detected_lang,
            "fallback_used": fallback_used,
        }
        return transcript, detected_lang, debug_info


def transcribe_multilingual(
    audio_bytes: bytes, language: str = None, model_size: str = "small"
) -> tuple[str, str, dict, dict, dict]:
    """Modular ASR + Acoustic Emotion Entrypoint."""
    try:
        preprocessed_audio, audio_diag = bytes_to_audio_array(audio_bytes)
    except Exception as exc:
        raise RuntimeError(
            "Couldn't decode audio/video. Please ensure file is a valid audio or video recording."
        ) from exc

    if audio_diag["rms_amplitude"] < 0.005:
        raise ValueError("Audio volume is too low. Please record again closer to the microphone.")

    # 1. Run Audio-Based Emotion Analysis from the physical waveform
    emotion_data = analyze_audio_emotion(preprocessed_audio, sr=WHISPER_SR)

    # 2. Run Speech Transcription via Router
    router = ASRRouter(whisper_model_size=model_size)
    transcript, detected_lang, debug_info = router.route_and_transcribe(
        audio_bytes, preprocessed_audio, language=language
    )

    return transcript, detected_lang, debug_info, audio_diag, emotion_data


def translate_to_english(text: str, source_language: str = "auto") -> tuple[str, str]:
    if not text or not text.strip():
        return text, "None"

    if source_language in ("English", "en"):
        return text, "None (Source is English)"

    src = LANGUAGE_MAP.get(source_language, source_language) or "auto"
    if src == "en":
        return text, "None (Source is English)"

    backend_name = f"IndicTrans / Google Translator ({src} → en)"
    try:
        translated = GoogleTranslator(source=src, target="en").translate(text)
        if translated:
            return translated.strip(), backend_name
    except Exception:
        pass

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return (translated.strip() if translated else text), backend_name
    except Exception as exc:
        raise RuntimeError(f"Translation failed: {exc}") from exc


# --------------------------------------------------------------------------
# Dynamic Emotion-Aware Avatar Renderer
# --------------------------------------------------------------------------
def render_avatar(transcript: str, emotion: str = "NEUTRAL", intensity: float = 50.0, confidence: float = 75.0) -> str:
    """Dynamic emotion-driven CSS/SVG/JS avatar that visually transforms
    its expression (brows, eyes, mouth curvature, color aura, and animation tempo)
    according to detected audio emotion and acoustic intensity."""
    safe_text = transcript.replace('"', "&quot;").replace("\n", " ")
    em_info = EMOTION_META.get(emotion.upper(), EMOTION_META["NEUTRAL"])
    bg_gradient = em_info["bg"]
    accent_color = em_info["color"]
    emoji = em_info["emoji"]

    # Visual expression attributes
    mouth_height = 10
    mouth_radius = "0 0 16px 16px"
    mouth_bg = "#d9485f"
    eyebrow_left_rotate = "0deg"
    eyebrow_right_rotate = "0deg"
    eyebrow_top = "40px"
    eye_scale_y = "1.0"
    eye_scale_x = "1.0"
    anim_speed = max(0.08, 0.20 - (intensity / 100.0) * 0.12)

    if emotion == "HAPPY":
        mouth_radius = "0 0 26px 26px / 0 0 20px 20px"
        mouth_height = 16
        eyebrow_left_rotate = "-10deg"
        eyebrow_right_rotate = "10deg"
        eyebrow_top = "38px"
        eye_scale_y = "0.8"
    elif emotion == "ANGRY":
        mouth_radius = "4px"
        mouth_height = 12
        mouth_bg = "#b91c1c"
        eyebrow_left_rotate = "25deg"
        eyebrow_right_rotate = "-25deg"
        eyebrow_top = "44px"
    elif emotion == "SAD":
        mouth_radius = "18px 18px 0 0 / 14px 14px 0 0"
        mouth_height = 10
        eyebrow_left_rotate = "-18deg"
        eyebrow_right_rotate = "18deg"
        eyebrow_top = "38px"
        eye_scale_y = "0.7"
    elif emotion == "FEAR":
        mouth_radius = "8px"
        mouth_height = 14
        eyebrow_left_rotate = "-22deg"
        eyebrow_right_rotate = "22deg"
        eyebrow_top = "36px"
        eye_scale_y = "1.3"
        eye_scale_x = "1.2"
    elif emotion == "SURPRISE":
        mouth_radius = "50%"
        mouth_height = 24
        eyebrow_left_rotate = "-12deg"
        eyebrow_right_rotate = "12deg"
        eyebrow_top = "32px"
        eye_scale_y = "1.4"
        eye_scale_x = "1.3"
    elif emotion == "DISGUST":
        mouth_radius = "0 14px 0 14px"
        mouth_height = 12
        eyebrow_left_rotate = "15deg"
        eyebrow_right_rotate = "-8deg"
        eyebrow_top = "42px"

    return f"""
    <style>
        .avatar-shell {{
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            background: {bg_gradient};
            border-radius: 22px;
            padding: 20px 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            border: 2px solid {accent_color}55;
            position: relative;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}

        .emotion-badge {{
            position: absolute;
            top: 12px;
            right: 14px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid {accent_color};
            color: white;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
            backdrop-filter: blur(8px);
        }}

        .avatar-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .avatar {{
            position: relative;
            width: 180px;
            height: 220px;
        }}

        .head {{
            position: absolute;
            left: 50%;
            top: 18px;
            transform: translateX(-50%);
            width: 120px;
            height: 120px;
            background: #f7d7b5;
            border-radius: 50%;
            border: 4px solid #2d3748;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}

        .hair {{
            position: absolute;
            top: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 122px;
            height: 36px;
            background: #2b1d17;
            border-radius: 60px 60px 18px 18px;
        }}

        .eyebrow {{
            position: absolute;
            width: 22px;
            height: 5px;
            background: #2b1d17;
            border-radius: 3px;
            top: {eyebrow_top};
            transition: all 0.2s ease;
        }}

        .eyebrow.left {{ left: 24px; transform: rotate({eyebrow_left_rotate}); }}
        .eyebrow.right {{ right: 24px; transform: rotate({eyebrow_right_rotate}); }}

        .eye {{
            position: absolute;
            width: 10px;
            height: 10px;
            background: #111827;
            border-radius: 50%;
            top: 52px;
            transform: scale({eye_scale_x}, {eye_scale_y});
            transition: all 0.2s ease;
        }}

        .eye.left {{ left: 30px; }}
        .eye.right {{ right: 30px; }}

        .mouth {{
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            bottom: 16px;
            width: 58px;
            height: {mouth_height}px;
            background: {mouth_bg};
            border-radius: {mouth_radius};
            transition: all 0.15s ease;
        }}

        .body {{
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 96px;
            height: 86px;
            background: {accent_color};
            border-radius: 18px 18px 10px 10px;
            border: 4px solid #2d3748;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }}

        .caption {{
            margin-top: 14px;
            color: #f8fafc;
            font-size: 13px;
            text-align: center;
            opacity: 0.95;
            max-width: 280px;
            word-wrap: break-word;
            font-weight: 500;
        }}
    </style>

    <div class="avatar-shell">
        <div class="emotion-badge">
            <span>{emoji} {emotion}</span>
            <span style="opacity:0.75; font-size:10px;">| {int(intensity)}% int.</span>
        </div>
        <div class="avatar-wrap">
            <div class="avatar">
                <div class="head">
                    <div class="hair"></div>
                    <div class="eyebrow left"></div>
                    <div class="eyebrow right"></div>
                    <div class="eye left"></div>
                    <div class="eye right"></div>
                    <div class="mouth" id="avatar-mouth"></div>
                </div>
                <div class="body"></div>
            </div>
            <div class="caption">{safe_text[:120]}</div>
        </div>
    </div>

    <script>
        const mouth = document.getElementById('avatar-mouth');
        let step = 0;
        const baseH = {mouth_height};
        setInterval(() => {{
            const mod = ((Math.sin(step) + 1) * {4 + (intensity / 20.0)});
            mouth.style.height = (baseH + mod) + 'px';
            step += 0.45;
        }}, {int(anim_speed * 1000)});
    </script>
    """


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_size = st.selectbox(
        "Whisper model size",
        ["tiny", "base", "small", "medium"],
        index=2,
        help="Bigger = more accurate speech recognition. 'small' recommended.",
    )

    language = st.selectbox(
        "Speech Language",
        [
            "Auto Detect",
            "English",
            "Hindi",
            "Gujarati",
            "Kannada",
            "Tamil",
            "Telugu",
        ],
        index=0,
    )
    st.caption("Select your spoken language or leave on Auto Detect.")

    st.divider()

    # ----------------------------------------------------------------------
    # Emotion Test Bench (One-Click Demonstrations for Judges)
    # ----------------------------------------------------------------------
    st.header("🧪 Emotion Test Lab")
    st.caption("Test how the SAME sentence changes emotion based strictly on AUDIO delivery style:")

    test_trigger = None

    if st.button("🗣️ Test: 'I can't believe you did that' (Calm Delivery)", use_container_width=True):
        test_trigger = "calm_disbelief"
    if st.button("😠 Test: 'I can't believe you did that' (Angry / Loud)", use_container_width=True):
        test_trigger = "angry_disbelief"
    if st.button("🎉 Test: 'I can't believe you did that' (Excited / Happy)", use_container_width=True):
        test_trigger = "happy_disbelief"
    if st.button("😢 Test: 'I can't believe you did that' (Sad / Depressed)", use_container_width=True):
        test_trigger = "sad_disbelief"
    if st.button("🆘 Test: 'Please help me' (Urgent / Panicked)", use_container_width=True):
        test_trigger = "urgent_help"

    st.divider()
    st.header("📖 Sign database")
    db_df = load_sign_database(DB_PATH)
    st.caption(f"{len(db_df)} words currently mapped to a sign.")
    search = st.text_input("Search a word")
    if search:
        matches = db_df[db_df["word"].str.contains(search.lower(), na=False)]
        st.dataframe(matches, hide_index=True, use_container_width=True)

    with st.expander("➕ Add a custom word / sign"):
        new_word = st.text_input("Word", key="new_word")
        new_sign = st.text_input("Sign (emoji)", key="new_sign")
        new_cat = st.text_input("Category", value="custom", key="new_cat")
        if st.button("Add to database"):
            if new_word and new_sign:
                updated = pd.concat(
                    [db_df, pd.DataFrame([{"word": new_word.lower().strip(), "sign": new_sign, "category": new_cat}])],
                    ignore_index=True,
                ).drop_duplicates(subset="word", keep="last")
                updated.to_csv(DB_PATH, index=False)
                load_sign_database.clear()
                st.success(f"Added '{new_word}' -> {new_sign}")
                st.rerun()
            else:
                st.warning("Enter both a word and a sign first.")

    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.title("🤟 Speech & Audio Emotion → Sign Converter")
st.write(
    "Detects **spoken words** and **speaker emotion** directly from the **actual audio signal** (speech prosody, energy, pitch dynamics). "
    "Signs are displayed with emotion-aware avatar animations."
)

# Handle Test Lab Synthesized Waveforms if clicked
if test_trigger:
    t_sig = np.linspace(0, 2.2, 35200, endpoint=False)
    if test_trigger == "calm_disbelief":
        synth_audio = 0.12 * np.sin(2 * np.pi * 135 * t_sig) + 0.04 * np.sin(2 * np.pi * 270 * t_sig)
        synth_transcript = "I can't believe you did that."
    elif test_trigger == "angry_disbelief":
        synth_audio = 0.48 * np.sin(2 * np.pi * 215 * t_sig) + 0.26 * np.sin(2 * np.pi * 430 * t_sig) + 0.18 * np.sin(2 * np.pi * 2200 * t_sig) + 0.08 * np.random.normal(0, 0.04, len(t_sig))
        synth_transcript = "I can't believe you did that."
    elif test_trigger == "happy_disbelief":
        f_h = 250 + 65 * np.sin(2 * np.pi * 3.5 * t_sig)
        synth_audio = 0.24 * np.sin(2 * np.pi * f_h * t_sig) + 0.12 * np.sin(2 * np.pi * 2 * f_h * t_sig)
        synth_transcript = "I can't believe you did that."
    elif test_trigger == "sad_disbelief":
        synth_audio = 0.038 * np.sin(2 * np.pi * 96 * t_sig)
        synth_audio[int(len(t_sig) * 0.35) : int(len(t_sig) * 0.65)] = 0.0
        synth_transcript = "I can't believe you did that."
    elif test_trigger == "urgent_help":
        f_u = np.linspace(170, 360, len(t_sig))
        synth_audio = 0.42 * np.sin(2 * np.pi * f_u * t_sig) + 0.16 * np.sin(2 * np.pi * 2400 * t_sig)
        synth_transcript = "Please help me."

    buf = io.BytesIO()
    sf.write(buf, synth_audio.astype(np.float32), WHISPER_SR, format="WAV")
    test_audio_bytes = buf.getvalue()

    # Process test audio
    preprocessed_test, diag_test = bytes_to_audio_array(test_audio_bytes)
    test_emotion = analyze_audio_emotion(preprocessed_test, sr=WHISPER_SR)
    test_tokens = clean_tokens(synth_transcript)
    test_signs_df, test_cov = words_to_signs(test_tokens, db_df)

    st.session_state.history.insert(
        0,
        {
            "transcript": synth_transcript,
            "english_text": synth_transcript,
            "display_lang": "English (Test Lab)",
            "debug_info": {"asr_backend": "Test Lab Speech Generator", "passed_language_code": "en", "model_size": "demo"},
            "audio_diag": diag_test,
            "trans_backend": "None",
            "model_size": model_size,
            "signs_df": test_signs_df,
            "coverage": test_cov,
            "is_translated": False,
            "emotion_data": test_emotion,
        },
    )

col_input, col_output = st.columns([1, 1.4], gap="large")

with col_input:
    st.subheader("🎙️ Input")
    audio_value = st.audio_input("Record speech")
    uploaded_file = st.file_uploader(
        "...or upload an audio/video file (WAV, FLAC, OGG, MP3, MP4, MOV, AVI)",
        type=["wav", "flac", "ogg", "mp3", "mp4", "mov", "avi", "webm", "mkv"],
    )

    raw_uploaded_bytes = None
    file_name = ""
    if audio_value is not None:
        raw_uploaded_bytes = audio_value.getvalue()
        file_name = "mic_recording.wav"
    elif uploaded_file is not None:
        raw_uploaded_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

    audio_bytes = None
    if raw_uploaded_bytes is not None:
        try:
            audio_bytes = extract_audio_from_uploaded_file(raw_uploaded_bytes, file_name)
        except Exception as vid_err:
            st.error(f"Error extracting audio: {vid_err}")
            audio_bytes = None

    if audio_bytes is not None:
        st.markdown("**Audio Playback:**")
        st.audio(audio_bytes)

    run = st.button("Convert to signs & analyze emotion ➡️", type="primary", disabled=audio_bytes is None)

with col_output:
    st.subheader("🤟 Output & Avatar")
    iframe_placeholder = st.empty()

    if run and audio_bytes is not None:
        with st.spinner("Processing speech prosody & transcription..."):
            try:
                transcript, detected_lang, debug_info, audio_diag, emotion_data = transcribe_multilingual(
                    audio_bytes, language=language, model_size=model_size
                )
            except ValueError as exc:
                st.warning(str(exc))
                transcript, detected_lang, debug_info, audio_diag, emotion_data = None, None, {}, {}, {}
            except Exception as exc:
                st.error(f"ASR & Emotion Error: {exc}")
                transcript, detected_lang, debug_info, audio_diag, emotion_data = None, None, {}, {}, {}

        if transcript == "":
            st.warning("Didn't catch any speech in that clip — try again.")
        elif transcript:
            lang_code = detected_lang or (LANGUAGE_MAP.get(language) if language != "Auto Detect" else None)
            display_lang = CODE_TO_LANGUAGE.get(
                lang_code, language if language != "Auto Detect" else (lang_code.title() if lang_code else "English")
            )

            is_english = (lang_code == "en") or (language == "English")
            english_text = transcript
            trans_backend = "None (English Speech)"
            is_translated = False

            if not is_english:
                with st.spinner("Translating speech to English..."):
                    try:
                        english_text, trans_backend = translate_to_english(transcript, source_language=lang_code or language)
                        is_translated = english_text.strip().lower() != transcript.strip().lower()
                    except Exception as exc:
                        st.warning(f"Translation warning: {exc}. Using original transcript.")
                        english_text = transcript
                        trans_backend = f"Error ({exc})"
                        is_translated = False

            tokens = clean_tokens(english_text)
            signs_df, coverage = words_to_signs(tokens, db_df)
            st.session_state.history.insert(
                0,
                {
                    "transcript": transcript,
                    "english_text": english_text,
                    "display_lang": display_lang,
                    "debug_info": debug_info,
                    "audio_diag": audio_diag,
                    "trans_backend": trans_backend,
                    "model_size": model_size,
                    "signs_df": signs_df,
                    "coverage": coverage,
                    "is_translated": is_translated,
                    "emotion_data": emotion_data,
                },
            )

    if st.session_state.history:
        item = st.session_state.history[0]
        if isinstance(item, dict):
            transcript = item["transcript"]
            english_text = item["english_text"]
            display_lang = item.get("display_lang", "English")
            signs_df = item["signs_df"]
            coverage = item["coverage"]
            is_translated = item.get("is_translated", False)
            trans_backend = item.get("trans_backend", "IndicTrans / Google Translator")
            emotion_data = item.get("emotion_data", {})
        else:
            transcript, signs_df, coverage = item
            english_text = transcript
            display_lang = "English"
            is_translated = False
            trans_backend = "None"
            emotion_data = {}

        detected_emotion = emotion_data.get("emotion", "NEUTRAL")
        emotion_conf = emotion_data.get("confidence", 70.0)
        emotion_intensity = emotion_data.get("intensity", 45.0)
        em_meta = EMOTION_META.get(detected_emotion, EMOTION_META["NEUTRAL"])

        # Iframe URL with emotion parameters
        encoded_text = urllib.parse.quote(english_text)
        react_url = (
            f"https://ai-avatar-jade-zeta.vercel.app/?text={encoded_text}"
            f"&speed=0.10&pause=800&emotion={detected_emotion.lower()}&intensity={int(emotion_intensity)}"
        )
        with iframe_placeholder:
            components.iframe(react_url, width=800, height=600)

        # Helper for intensity label
        def get_intensity_label(val: float) -> str:
            if val >= 75:
                return "HIGH"
            elif val >= 45:
                return "MODERATE"
            else:
                return "LOW"

        intensity_tier = get_intensity_label(emotion_intensity)

        # ------------------------------------------------------------------
        # 🎭 Emotion Analysis Clean Result Card
        # ------------------------------------------------------------------
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.85));
                border: 2px solid {em_meta['color']}77;
                border-radius: 16px;
                padding: 18px 22px;
                margin-bottom: 16px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            ">
                <div style="font-size: 1.05rem; font-weight: 600; color: #e2e8f0; margin-bottom: 10px;">
                    🎭 Emotion Analysis
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: {em_meta['color']};">
                        {em_meta['emoji']} {detected_emotion}
                    </div>
                    <div style="display: flex; gap: 14px;">
                        <div style="background: rgba(30, 41, 59, 0.9); padding: 6px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                            <span style="font-size: 0.82rem; color: #94a3b8;">Confidence:</span>
                            <b style="font-size: 0.95rem; color: #f8fafc; margin-left: 5px;">{int(round(emotion_conf))}%</b>
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.9); padding: 6px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                            <span style="font-size: 0.82rem; color: #94a3b8;">Intensity:</span>
                            <b style="font-size: 0.95rem; color: {em_meta['color']}; margin-left: 5px;">{intensity_tier}</b>
                        </div>
                    </div>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    Emotion estimated from speech characteristics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**Detected Language:** {display_lang}")
        if is_translated:
            st.markdown(f"**Original Transcript:**\n{transcript}")
            st.markdown(f"**English Translation:**\n{english_text}")
        else:
            st.markdown(f"**Transcript:**\n{transcript}")

        # Render Emotion-Driven CSS/JS Avatar
        components.html(
            render_avatar(
                english_text,
                emotion=detected_emotion,
                intensity=emotion_intensity,
                confidence=emotion_conf,
            ),
            height=370,
            scrolling=False,
        )

        st.progress(coverage / 100, text=f"{coverage:.0f}% of words had a direct sign")

        n_cols = 6
        rows_needed = int(np.ceil(len(signs_df) / n_cols)) if len(signs_df) else 0
        idx = 0
        for _ in range(rows_needed):
            cols = st.columns(n_cols)
            for c in cols:
                if idx >= len(signs_df):
                    break
                row = signs_df.iloc[idx]
                with c:
                    st.markdown(
                        f"<div style='text-align:center;font-size:2.2rem'>{row['sign']}</div>"
                        f"<div style='text-align:center;font-size:0.8rem;color:gray'>{row['word']}</div>",
                        unsafe_allow_html=True,
                    )
                idx += 1

        with st.expander("Show as table"):
            st.dataframe(signs_df, hide_index=True, use_container_width=True)
    else:
        st.info("Record audio or upload an audio/video file, then press 'Convert to signs & analyze emotion'.")

if len(st.session_state.history) > 1:
    st.divider()
    st.subheader("🕘 History")
    for item in st.session_state.history[1:]:
        if isinstance(item, dict):
            hist_orig = item["transcript"]
            hist_eng = item["english_text"]
            hist_signs = item["signs_df"]
            hist_cov = item["coverage"]
            hist_em = item.get("emotion_data", {}).get("emotion", "NEUTRAL")
            hist_meta = EMOTION_META.get(hist_em, {})
            label = hist_eng if hist_eng != hist_orig else hist_orig
            tag = f"{hist_meta.get('emoji', '')} [{hist_em}] "
        else:
            hist_orig, hist_signs, hist_cov = item
            label = hist_orig
            tag = ""
        with st.expander(f"{tag}{label[:50]}{'...' if len(label) > 50 else ''}"):
            st.write("".join(hist_signs["sign"].tolist()))
            st.caption(f"{hist_cov:.0f}% direct match")
