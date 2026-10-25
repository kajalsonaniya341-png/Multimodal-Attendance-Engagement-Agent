"""
Acoustic Interaction & Noise Dynamics Module
EPIC 3: Audio Speech & Noise Processing
Privacy-First Design: Local Edge Spectral Processing (No Cloud Audio Streaming)
Platform ID: 6ac775fb32e2e6be34c17468
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


class AcousticState:
    SILENCE = "Silence"
    LECTURE = "Instructor Lecture"
    DISCUSSION = "Student Discussion / Verbal Interaction"
    DISTRACTION = "High Ambient Classroom Distraction"


class AcousticAnalyzer:
    """
    Acoustic signal processor that segments audio into time windows and extracts:
    - RMS Energy (Decibels)
    - Zero-Crossing Rate (ZCR)
    - Spectral Centroid
    - Voice Activity Detection (VAD) profile
    Classifies audio segments into 4 pedagogical classroom states:
    1. Silence
    2. Instructor Lecture
    3. Student Discussion / Verbal Interaction
    4. High Ambient Classroom Distraction
    """

    def __init__(self, sample_rate: int = 16000, window_sec: float = 1.0, hop_sec: float = 1.0):
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.window_samples = int(sample_rate * window_sec)
        self.hop_samples = int(sample_rate * hop_sec)

        # Configurable calibration thresholds
        self.silence_rms_threshold_db = -42.0  # dBFS below which is considered silent
        self.distraction_rms_threshold_db = -12.0  # loud commotion or chatter
        self.distraction_centroid_hz = 3200.0   # high-frequency clatter/shuffling

    def compute_rms(self, samples: np.ndarray) -> Tuple[float, float]:
        """Compute RMS amplitude and decibels (dBFS)."""
        if len(samples) == 0:
            return 0.0, -100.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        # Convert to dBFS (reference 1.0 for float samples)
        db = 20.0 * math.log10(max(1e-5, rms))
        return rms, max(-100.0, db)

    def compute_zcr(self, samples: np.ndarray) -> float:
        """Compute Zero Crossing Rate for speech/fricative dynamics."""
        if len(samples) < 2:
            return 0.0
        signs = np.sign(samples)
        # Replace 0 with 1 to avoid zero-division issues
        signs[signs == 0] = 1
        zero_crossings = np.sum(np.abs(np.diff(signs))) / (2.0 * len(samples))
        return float(zero_crossings)

    def compute_spectral_centroid(self, samples: np.ndarray) -> float:
        """Compute spectral centroid (perceived spectral brightness/formant center)."""
        if len(samples) < 256:
            return 1000.0
        # Window with Hann to reduce spectral leakage
        windowed = samples * np.hanning(len(samples))
        fft_mags = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self.sample_rate)

        sum_mags = np.sum(fft_mags)
        if sum_mags < 1e-6:
            return 0.0
        centroid = float(np.sum(freqs * fft_mags) / sum_mags)
        return centroid

    def classify_window(
        self,
        rms_db: float,
        zcr: float,
        centroid: float,
        recent_energy_variance: float = 0.0
    ) -> Dict[str, Any]:
        """
        Classifies single acoustic window into one of 4 classroom acoustic states:
        - Silence: very low energy, contemplation or independent study
        - Instructor Lecture: sustained single-source vocal formant, moderate energy, stable envelope
        - Student Discussion: dynamic multi-speaker vocal energy, conversational turn-taking
        - High Ambient Distraction: high noise floor, furniture clatter, loud commotion
        """
        vad_active = rms_db > self.silence_rms_threshold_db

        # 1. Check for silence / quiet study
        if not vad_active:
            state = AcousticState.SILENCE
            confidence = min(1.0, max(0.5, (self.silence_rms_threshold_db - rms_db) / 20.0))
            engagement_weight = 65.0  # Contemplative/neutral

        # 2. Check for high ambient distraction / commotion
        elif rms_db > self.distraction_rms_threshold_db or (rms_db > -20.0 and centroid > self.distraction_centroid_hz):
            state = AcousticState.DISTRACTION
            confidence = 0.88
            engagement_weight = 25.0  # Disruptive

        # 3. Check for student discussion vs instructor lecture
        # Speech range: centroid typically 600 - 2800 Hz.
        # Multi-speaker conversational discussion exhibits higher dynamic variance and burstiness.
        elif recent_energy_variance > 18.0 or (zcr > 0.12 and -32.0 < rms_db < -14.0):
            state = AcousticState.DISCUSSION
            confidence = 0.82
            engagement_weight = 92.0  # Highly active participation

        else:
            state = AcousticState.LECTURE
            confidence = 0.85
            engagement_weight = 80.0  # Attentive instructional reception

        return {
            "state": state,
            "confidence": round(confidence, 2),
            "vad_active": vad_active,
            "engagement_weight": engagement_weight
        }

    def process_audio_buffer(self, audio_data: np.ndarray, sample_rate: Optional[int] = None) -> pd.DataFrame:
        """
        Segments a 1D audio waveform into time windows and extracts telemetry time series.
        """
        if sample_rate and sample_rate != self.sample_rate:
            # Resample if needed using linear interpolation
            num_target_samples = int(len(audio_data) * self.sample_rate / sample_rate)
            x_old = np.linspace(0, 1, len(audio_data))
            x_new = np.linspace(0, 1, num_target_samples)
            audio_data = np.interp(x_new, x_old, audio_data)

        # Ensure float32 normalized between -1.0 and 1.0
        if audio_data.dtype in (np.int16, np.int32):
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        elif np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)

        records = []
        recent_dbs = []

        total_samples = len(audio_data)
        if total_samples < self.window_samples:
            # Pad short audio
            audio_data = np.pad(audio_data, (0, self.window_samples - total_samples))
            total_samples = len(audio_data)

        for start_idx in range(0, total_samples - self.window_samples + 1, self.hop_samples):
            window = audio_data[start_idx:start_idx + self.window_samples]
            timestamp = start_idx / self.sample_rate

            rms, db = self.compute_rms(window)
            zcr = self.compute_zcr(window)
            centroid = self.compute_spectral_centroid(window)

            recent_dbs.append(db)
            if len(recent_dbs) > 6:
                recent_dbs.pop(0)
            energy_variance = float(np.var(recent_dbs)) if len(recent_dbs) > 2 else 0.0

            classification = self.classify_window(db, zcr, centroid, energy_variance)

            records.append({
                "timestamp_sec": round(timestamp, 2),
                "rms_amplitude": round(rms, 4),
                "decibel_db": round(db, 1),
                "zcr": round(zcr, 4),
                "spectral_centroid_hz": round(centroid, 1),
                "vad_active": classification["vad_active"],
                "acoustic_state": classification["state"],
                "confidence": classification["confidence"],
                "acoustic_engagement_factor": classification["engagement_weight"]
            })

        return pd.DataFrame(records)

    def generate_synthetic_classroom_audio(self, duration_sec: int = 120) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generates realistic multi-phase classroom acoustic signal:
        - 0-25s: Instructor Lecture (focused vocal frequencies ~150-2500Hz)
        - 25-50s: Student Discussion & Q&A (multi-pitch conversational bursts)
        - 50-70s: High Ambient Distraction (spikes, rustling, chatter)
        - 70-90s: Silence / Silent Reading / Individual Practice
        - 90-120s: Instructor Synthesis Lecture
        Returns raw audio waveform and analyzed DataFrame.
        """
        sr = self.sample_rate
        total_samples = int(sr * duration_sec)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)
        audio = np.zeros(total_samples, dtype=np.float32)

        # Baseline gentle ambient room tone (HVAC / room acoustics)
        audio += 0.015 * np.random.normal(0, 1, total_samples)

        for idx, sec in enumerate(t):
            # Phase 1: Instructor Lecture (0-25s)
            if 0 <= sec < 25:
                # Modulated harmonic vocal formants
                vocal = (
                    0.20 * np.sin(2 * np.pi * 180 * sec) +
                    0.15 * np.sin(2 * np.pi * 540 * sec) +
                    0.10 * np.sin(2 * np.pi * 1200 * sec)
                )
                # Speech cadence pause envelope
                cadence = 0.5 + 0.5 * np.sin(2 * np.pi * 1.8 * sec)
                audio[idx] += vocal * cadence

            # Phase 2: Student Discussion / Verbal Interaction (25-55s)
            elif 25 <= sec < 55:
                # Multiple overlapping speakers with dynamic turn-taking
                s1 = 0.18 * np.sin(2 * np.pi * 220 * sec) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.2 * sec))
                s2 = 0.16 * np.sin(2 * np.pi * 340 * sec) * (0.5 + 0.5 * np.cos(2 * np.pi * 2.7 * sec))
                audio[idx] += s1 + s2 + 0.03 * np.random.normal(0, 1)

            # Phase 3: High Ambient Classroom Distraction (55-75s)
            elif 55 <= sec < 75:
                # High energy bursty noise (shuffling chairs, loud murmur)
                commotion = 0.35 * np.random.normal(0, 1)
                periodic_spike = 0.40 * (1.0 if (int(sec * 4) % 7 == 0) else 0.0)
                audio[idx] += commotion + periodic_spike

            # Phase 4: Silence / Individual Reflection (75-95s)
            elif 75 <= sec < 95:
                # Extremely quiet baseline
                audio[idx] += 0.005 * np.random.normal(0, 1)

            # Phase 5: Closing Instructor Lecture (95-120s)
            else:
                vocal = 0.18 * np.sin(2 * np.pi * 175 * sec) + 0.12 * np.sin(2 * np.pi * 600 * sec)
                audio[idx] += vocal * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * sec))

        # Normalize to avoid clipping
        audio = audio / np.max(np.abs(audio) + 1e-5) * 0.85
        df = self.process_audio_buffer(audio, sr)
        return audio, df
