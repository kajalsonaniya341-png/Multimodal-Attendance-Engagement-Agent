"""
Unit and Integration Test Suite
Multimodal Attendance & Engagement Agent
Skill Wallet Capstone Project 5 | Platform ID: 6ac775fb32e2e6be34c17468
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

# Add app directory to sys.path
app_dir = Path(__file__).resolve().parent.parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from presence_tracker import PresenceTracker, FaceTrack
from acoustic_analyzer import AcousticAnalyzer, AcousticState
from engagement_fusion import EngagementFusionEngine
from teaching_advisor import TeachingAdvisorEngine
from session_analytics import SessionAnalyticsEngine
from synthetic_generator import SyntheticClassroomGenerator


class TestEpic2PresenceTracking(unittest.TestCase):
    """EPIC 2: Face Tracking & Sustained Presence Attendance (>70% requirement)."""

    def setUp(self):
        self.tracker = PresenceTracker(sustained_threshold=0.70)

    def test_sustained_presence_logic(self):
        """Verify student present for >70% is marked Present, while <70% is marked Absent/Transient."""
        session_duration = 100.0  # seconds

        # Student 1: Present for 85 seconds (85% > 70%) -> Present
        t1 = FaceTrack(track_id=1, initial_box=(50, 50, 60, 60), timestamp=0.0)
        t1.observed_seconds = 85.0
        t1.last_seen = 85.0
        self.tracker.active_tracks[1] = t1

        # Student 2: Transient hallway passerby, seen for only 15 seconds (15% < 70%) -> Absent
        t2 = FaceTrack(track_id=2, initial_box=(200, 50, 60, 60), timestamp=30.0)
        t2.observed_seconds = 15.0
        t2.last_seen = 45.0
        self.tracker.active_tracks[2] = t2

        register = self.tracker.get_attendance_register(total_session_duration=session_duration)

        self.assertEqual(len(register), 2)
        s1_row = register[register["Student ID"] == "STUDENT_001"].iloc[0]
        s2_row = register[register["Student ID"] == "STUDENT_002"].iloc[0]

        self.assertEqual(s1_row["Status"], "Present")
        self.assertIn("Absent", s2_row["Status"])
        self.assertAlmostEqual(s1_row["Presence Rate (%)"], 85.0, delta=1.0)
        self.assertAlmostEqual(s2_row["Presence Rate (%)"], 15.0, delta=1.0)

    def test_anonymized_id_format(self):
        """Verify participants receive ethical anonymized tokens STUDENT_001, etc."""
        track = FaceTrack(track_id=7, initial_box=(10, 10, 50, 50), timestamp=0.0)
        self.assertEqual(track.anonymized_id, "STUDENT_007")

    def test_head_pose_estimation(self):
        """Verify Euler angle calculation and focus scoring."""
        # Simulated 2D landmarks for forward face
        pts = np.array([
            (320.0, 240.0),  # Nose
            (320.0, 310.0),  # Chin
            (270.0, 210.0),  # Left eye
            (370.0, 210.0),  # Right eye
            (290.0, 280.0),  # Left mouth
            (350.0, 280.0)   # Right mouth
        ], dtype=np.float64)

        pose = self.tracker._estimate_head_pose(pts, 640, 480)
        self.assertIn("yaw", pose)
        self.assertIn("pitch", pose)
        self.assertIn("focus_score", pose)
        self.assertGreaterEqual(pose["focus_score"], 0.0)
        self.assertLessEqual(pose["focus_score"], 100.0)


class TestEpic3AcousticProcessing(unittest.TestCase):
    """EPIC 3: Audio Speech & Noise Processing (Silence, Lecture, Discussion, Distraction)."""

    def setUp(self):
        self.analyzer = AcousticAnalyzer(sample_rate=16000, window_sec=1.0)

    def test_silence_classification(self):
        """Very low energy signals must be classified as Silence."""
        silent_audio = np.zeros(16000, dtype=np.float32)
        rms, db = self.analyzer.compute_rms(silent_audio)
        zcr = self.analyzer.compute_zcr(silent_audio)
        centroid = self.analyzer.compute_spectral_centroid(silent_audio)

        res = self.analyzer.classify_window(db, zcr, centroid)
        self.assertEqual(res["state"], AcousticState.SILENCE)
        self.assertFalse(res["vad_active"])

    def test_distraction_classification(self):
        """High volume erratic noise must be classified as High Ambient Classroom Distraction."""
        res = self.analyzer.classify_window(rms_db=-8.0, zcr=0.25, centroid=3800.0)
        self.assertEqual(res["state"], AcousticState.DISTRACTION)

    def test_discussion_classification(self):
        """High dynamic energy variance must be classified as Student Discussion."""
        res = self.analyzer.classify_window(rms_db=-22.0, zcr=0.14, centroid=1800.0, recent_energy_variance=25.0)
        self.assertEqual(res["state"], AcousticState.DISCUSSION)

    def test_synthetic_audio_generation(self):
        """Verify synthetic audio signal and dataframe structure."""
        audio, df = self.analyzer.generate_synthetic_classroom_audio(duration_sec=10)
        self.assertEqual(len(audio), 16000 * 10)
        self.assertFalse(df.empty)
        self.assertIn("acoustic_state", df.columns)
        self.assertIn("decibel_db", df.columns)


class TestEpic4MultimodalFusion(unittest.TestCase):
    """EPIC 4: Multimodal Signal Fusion & Engagement Scoring."""

    def setUp(self):
        self.engine = EngagementFusionEngine(visual_weight=0.60, audio_weight=0.40)

    def test_engagement_normalization(self):
        """Score must always be bounded between 0 and 100."""
        score, zone, _ = self.engine.compute_instantaneous_engagement(
            visual_score=95.0, acoustic_score=90.0, acoustic_state="Student Discussion / Verbal Interaction"
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertEqual(zone, "High Engagement")

    def test_distraction_penalty(self):
        """High distraction state should apply downward penalty."""
        normal_score, _, _ = self.engine.compute_instantaneous_engagement(
            visual_score=70.0, acoustic_score=70.0, acoustic_state="Instructor Lecture"
        )
        distracted_score, _, _ = self.engine.compute_instantaneous_engagement(
            visual_score=70.0, acoustic_score=70.0, acoustic_state="High Ambient Classroom Distraction"
        )
        self.assertGreater(normal_score, distracted_score)

    def test_peaks_and_dips_detection(self):
        """Verify detection of attention peaks and dips."""
        gen = SyntheticClassroomGenerator(duration_sec=30)
        _, tel_df, _ = gen.generate_synthetic_session_telemetry()
        extremes = self.engine.detect_peaks_and_dips(tel_df)
        self.assertIn("peaks", extremes)
        self.assertIn("dips", extremes)
        self.assertIn("summary", extremes)


class TestEpic5TeachingAdvisor(unittest.TestCase):
    """EPIC 5: AI Teaching Suggestion Engine."""

    def setUp(self):
        self.advisor = TeachingAdvisorEngine(lull_threshold=45.0, high_threshold=70.0)

    def test_attention_lull_trigger(self):
        """Prolonged low engagement must trigger actionable quick poll / stretch break prompt."""
        # Simulated stream with engagement < 45 for 10 consecutive seconds
        rows = []
        for t in range(10):
            rows.append({
                "timestamp_sec": float(t),
                "fused_engagement_index": 38.0,
                "acoustic_state": "Instructor Lecture",
                "visual_attentiveness": 35.0
            })
        df = pd.DataFrame(rows)
        advisories = self.advisor.evaluate_telemetry_stream(df)

        self.assertTrue(any("poll" in a.recommendation.lower() or "stretch" in a.recommendation.lower() for a in advisories))

    def test_offline_pedagogical_assistant(self):
        """Verify heuristic assistant responds to teacher queries."""
        response = self.advisor.answer_pedagogical_query("How do I conduct a quick poll?")
        self.assertIn("poll", response.lower())


class TestEpic6SessionAnalytics(unittest.TestCase):
    """EPIC 5 & 6: Session Analytics, KPIs, and JSON/Markdown Reporting."""

    def setUp(self):
        self.analytics = SessionAnalyticsEngine(platform_id="6ac775fb32e2e6be34c17468")

    def test_session_kpis_and_json_export(self):
        gen = SyntheticClassroomGenerator(duration_sec=30)
        att_df, tel_df, _ = gen.generate_synthetic_session_telemetry()

        kpis = self.analytics.compute_session_kpis(att_df, tel_df)
        self.assertIn("average_engagement_score", kpis)
        self.assertIn("attendance_rate_pct", kpis)
        self.assertIn("total_students_detected", kpis)

        json_out = self.analytics.generate_json_report(kpis, att_df, tel_df, [])
        self.assertIn("6ac775fb32e2e6be34c17468", json_out)
        self.assertIn("summary_kpis", json_out)

        md_out = self.analytics.generate_markdown_report(kpis, [])
        self.assertIn("Educator Post-Session Intelligence Report", md_out)
        self.assertIn("6ac775fb32e2e6be34c17468", md_out)


if __name__ == "__main__":
    unittest.main()
