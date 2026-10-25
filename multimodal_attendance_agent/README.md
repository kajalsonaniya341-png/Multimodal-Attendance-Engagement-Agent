# Multimodal Attendance & Engagement Agent
**Skill Wallet Capstone Project 5 | Category: Generative AI & Privacy-First Classroom Analytics**  
**Platform ID:** `6ac775fb32e2e6be34c17468`

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg)](https://streamlit.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh%20%26%20Pose-007FFF.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![Privacy-First](https://img.shields.io/badge/Privacy-100%25%20Offline%20Edge-059669.svg)](#1-privacy-first-system-principles)
[![Tests](https://img.shields.io/badge/Tests-13%2F13%20Passing-brightgreen.svg)](#testing--verification)

---

## 🎯 Executive Overview & Scope

The **Multimodal Attendance & Engagement Agent** is a production-ready, privacy-preserving classroom intelligence system. It fuses **computer vision** (MediaPipe facial landmark tracking and head pose orientation) and **acoustic signal processing** (spectral centroid, zero-crossing rate, RMS energy, and voice activity detection) to:
1. **Automate Sustained Presence Attendance:** Prevent false attendance marks from temporary hallway pass-throughs by enforcing a **>70% sustained duration threshold**.
2. **Classify Acoustic Classroom Dynamics:** Distinguish in real time between **Silence**, **Instructor Lecture**, **Student Discussion / Verbal Interaction**, and **High Ambient Classroom Distraction**.
3. **Compute Real-Time Multimodal Engagement (0–100):** Generate synchronized telemetry curves showing attention peaks, plateaus, and cognitive lulls.
4. **Trigger AI Teaching Advisories:** Provide heuristic pedagogical prompts (e.g., launching quick pulse polls, think-pair-share, or stretch breaks when collective focus dips below 45%).
5. **Generate Diagnostic Session Reports:** Export complete audit reports in **CSV**, **JSON**, and printable **Executive Markdown**.

---

## 🛡️ 1. Privacy-First System Principles

- **100% Offline Edge Processing:** Zero video or audio data is ever transmitted to the cloud. All face tracking, spectral decomposition, and heuristic advice algorithms execute locally on-device.
- **Ethical Anonymized Telemetry:** Focuses on collective pedagogical feedback rather than punitive individual surveillance.
- **Anonymous Tracking Tokens:** Participants are tracked strictly via anonymous tokens (`STUDENT_001`, `STUDENT_002`, ...); no biometric templates, names, or face embeddings are stored.
- **On-Device Face Blur:** Integrated privacy shield obfuscates facial pixels in visual displays while preserving pose orientation vectors.

---

## 📐 System Architecture & Epics Coverage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INPUT FEEDS (OFFLINE EDGE)                            │
│   Webcam / Camera Feed     Classroom Video (MP4/AVI)     Classroom Audio    │
└──────────────────────┬──────────────────────────────────┬───────────────────┘
                       │                                  │
       ┌───────────────▼────────────────┐ ┌───────────────▼─────────────────┐
       │     EPIC 2: COMPUTER VISION    │ │    EPIC 3: ACOUSTIC DSP         │
       │  • MediaPipe Face Mesh / Pose  │ │  • Windowed Segmentation (1.0s) │
       │  • Head Pose (Yaw, Pitch, Roll)│ │  • RMS Energy / dBFS Level      │
       │  • Centroid + IoU Tracking     │ │  • Zero-Crossing Rate (ZCR)     │
       │  • Sustained Duration (>70%)   │ │  • Spectral Centroid Formants   │
       │  • Pass-Through Rejection      │ │  • 4-Class Acoustic Classifier  │
       └───────────────┬────────────────┘ └───────────────┬─────────────────┘
                       │                                  │
                       └────────────────┬─────────────────┘
                                        ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │             EPIC 4: MULTIMODAL FUSION & SCORING                    │
       │   Formula: E(t) = clip(wv*V(t) + wa*A(t) + Synergy - Penalty)      │
       │   Synchronized Time-Step Matrix & Interactive Plotly Curves        │
       │   Engagement Zones: High (>70), Moderate (45-70), Lull (<45)       │
       └────────────────────────────────┬───────────────────────────────────┘
                                        ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │             EPIC 5: HEURISTIC TEACHING ADVISOR                     │
       │   • Collective Engagement Dip (<45% for 30s) -> 60s Pulse Poll     │
       │   • Extended Monologue (>15 min) -> Think-Pair-Share Pivot         │
       │   • High Ambient Noise -> Auditory Reset Cue / Clarify Task        │
       │   • Interactive Pedagogical Assistant Simulator                    │
       └────────────────────────────────┬───────────────────────────────────┘
                                        ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │             EPIC 6: TELEMETRY DASHBOARD & REPORTS                  │
       │   • Modern 5-Tab Educational Streamlit Interface                   │
       │   • Download Attendance Register (CSV)                             │
       │   • Download Multimodal Telemetry Matrix (CSV)                     │
       │   • Download Session Diagnostic Audit Report (JSON)                │
       │   • Download Executive Educator Diagnostic Report (Markdown)       │
       └────────────────────────────────────────────────────────────────────┘
```

---

## 🧮 Multimodal Mathematical Formulation

The unified **Engagement Index** $E(t) \in [0, 100]$ at time $t$ is computed as:

$$E(t) = \text{clip}\Big( w_v \cdot V(t) + w_a \cdot A(t) + \text{Synergy}(V, A) - \text{DistractionPenalty}, 0, 100 \Big)$$

Where:
- $V(t) \in [0, 100]$: Collective visual focus score derived from facial landmark Euler angles:
  $$\text{Pitch penalty} = \max(0, |\text{pitch}| - 15^\circ) \times 1.8$$
  $$\text{Yaw penalty} = \max(0, |\text{yaw}| - 15^\circ) \times 1.5$$
- $A(t) \in [0, 100]$: Pedagogical acoustic interaction factor based on classified state:
  - **Student Discussion / Verbal Interaction:** $92.0$ (High active learning)
  - **Instructor Lecture:** $80.0$ (Attentive instructional reception)
  - **Silence / Quiet Reflection:** $65.0$ (Contemplative / study baseline)
  - **High Ambient Classroom Distraction:** $25.0$ (Disruptive noise floor)
- $w_v, w_a$: Configurable weights (default: $w_v = 0.60, w_a = 0.40$).
- $\text{Synergy}(V, A)$:
  - **$+6.0$ Synergy Boost:** When high student discussion coincides with forward visual alignment.
  - **$-14.0$ Distraction Penalty:** When ambient room chatter surges and visual attention fragments.

---

## 🗂️ Application Modules (5-Tab Structure)

### Tab 1: Automated Presence-Based Attendance (Epic 2)
- Video processor supporting live webcam, uploaded video, or built-in test drive.
- OpenCV & MediaPipe Face Mesh tracking with persistent anonymized IDs (`STUDENT_001`, `STUDENT_002`, ...).
- **Sustained Presence Logic:** Requires participants to be observed for **>70%** of total session duration. Transient visitors (e.g. hallway pass-throughs seen for 10–15 seconds) are automatically categorized as `Absent (Transient / <70%)`.
- Exportable attendance register table with **CSV download**.

### Tab 2: Acoustic Interaction & Noise Dynamics (Epic 3)
- Real-time windowed acoustic signal processor (1.0s hop).
- Spectral energy, ZCR, and VAD classifying audio into:
  1. **Silence**
  2. **Instructor Lecture**
  3. **Student Discussion / Verbal Interaction**
  4. **High Ambient Classroom Distraction**
- Acoustic decibel timeline, spectrogram dynamics, and distribution donut charts.

### Tab 3: Multimodal Fusion & Engagement Scoring (Epic 4)
- Fuses visual attentiveness indicators (forward head pose, upright posture vs slumping) and acoustic interaction energy into a unified time-step matrix.
- Interactive Plotly engagement curves with green ($\ge 70$), yellow ($45–70$), and red ($< 45$) zones.
- Automatic detection of attention peaks and cognitive dips.

### Tab 4: AI Teaching Suggestion Engine (Epic 5)
- Rule-based heuristic advisory agent evaluating temporal attention trends.
- Automated pedagogical alerts:
  - *“Collective engagement dropped below 45% — recommend launching a quick poll, interactive discussion, or a 2-minute stretch break.”*
  - *“Extended monologue detected — pivot to think-pair-share concept check.”*
  - *“High ambient chatter detected — execute auditory reset cue.”*
- Interactive pedagogical chat assistant for educator advice.

### Tab 5: Session Analytics & Educator Report (Epic 5 & 6)
- Comprehensive post-session KPI summary: average engagement, sustained attendance rate, peak attention time, lecture vs discussion ratio.
- 1-Click Export Center:
  - `attendance_register_6ac775fb32e2e6be34c17468.csv`
  - `multimodal_telemetry_6ac775fb32e2e6be34c17468.csv`
  - `session_audit_6ac775fb32e2e6be34c17468.json`
  - `educator_report_6ac775fb32e2e6be34c17468.md`

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Windows, macOS, or Linux

### 1. Clone or Extract Project
```bash
git clone https://github.com/your-username/multimodal_attendance_agent.git
cd multimodal_attendance_agent
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser. The application loads with a pre-configured 1-click classroom demonstration session ready for instant evaluation!

---

## 🧪 Testing & Verification

A comprehensive test suite is included in `tests/test_agent.py` covering all 5 epics:
```bash
python -m unittest multimodal_attendance_agent/tests/test_agent.py
```

### Test Suite Execution Output:
```
.............
----------------------------------------------------------------------
Ran 13 tests in 2.861s

OK
```
All 13 unit tests pass, verifying:
- Sustained presence attendance thresholding (>70% vs transient hallway discard)
- Anonymized token assignment
- Head pose Euler angle estimation
- 4-class acoustic state classification (Silence, Lecture, Discussion, Distraction)
- Multimodal mathematical normalization and distraction penalty
- Attention lull trigger (<45%) and pedagogical advisory generation
- JSON and Markdown report generation with Platform ID `6ac775fb32e2e6be34c17468`

---

## ☁️ 1-Click Deployment (Streamlit Community Cloud)

This repository is engineered with **zero external database dependencies** (in-memory state, local file exports, and self-contained models).

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**, select your repository, branch `main`, and main file path `app.py`.
4. Click **"Deploy"**!

---

## 📄 License & Ethical Compliance

Developed under Privacy-First Educational AI guidelines. Zero biometrics or video streams are retained. For educational research and pedagogical formative assessment only.
Platform ID: `6ac775fb32e2e6be34c17468`
