"""
Multimodal Signal Fusion & Engagement Scoring Module
EPIC 4: Multimodal Signal Fusion & Engagement Scoring
Platform ID: 6ac775fb32e2e6be34c17468
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


class EngagementFusionEngine:
    """
    Fuses visual attentiveness indicators (forward head pose, posture vs slumping)
    and audio interaction energy (lecture, discussion, silence, distraction)
    into a unified time-step telemetry matrix and normalized Engagement Index (0-100).
    """

    def __init__(
        self,
        visual_weight: float = 0.60,
        audio_weight: float = 0.40,
        high_threshold: float = 70.0,
        lull_threshold: float = 45.0
    ):
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.high_threshold = high_threshold
        self.lull_threshold = lull_threshold

    def compute_instantaneous_engagement(
        self,
        visual_score: float,
        acoustic_score: float,
        acoustic_state: str,
        active_participants: int = 1
    ) -> Tuple[float, str, float]:
        """
        Computes single timestep engagement index (0-100) with pedagogical synergy adjustments.
        Returns: (fused_score, engagement_zone, synergy_delta)
        """
        # Baseline weighted combination
        raw_score = (self.visual_weight * visual_score) + (self.audio_weight * acoustic_score)

        # Contextual pedagogical synergy adjustments
        synergy = 0.0

        if acoustic_state == "Student Discussion / Verbal Interaction" and visual_score >= 65.0:
            # Active collaborative problem solving
            synergy += 6.0
        elif acoustic_state == "High Ambient Classroom Distraction":
            # Environmental chaos penalty
            synergy -= 14.0
        elif acoustic_state == "Silence" and visual_score < 50.0:
            # Silence + head down = collective disengagement/drowsiness
            synergy -= 10.0
        elif acoustic_state == "Instructor Lecture" and visual_score >= 75.0:
            # High instructional focus
            synergy += 4.0

        fused_score = float(np.clip(raw_score + synergy, 0.0, 100.0))

        # Determine engagement zone
        if fused_score >= self.high_threshold:
            zone = "High Engagement"
        elif fused_score >= self.lull_threshold:
            zone = "Moderate / Passive Focus"
        else:
            zone = "Attention Deficit / Lull"

        return round(fused_score, 1), zone, round(synergy, 1)

    def fuse_telemetry_streams(
        self,
        visual_records: List[Dict[str, Any]],
        audio_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aligns visual frame time series with acoustic windows into a synchronized
        time-step matrix.
        """
        if audio_df is None or audio_df.empty:
            timestamps = [r.get("timestamp_sec", i) for i, r in enumerate(visual_records)] if visual_records else list(range(60))
            audio_df = pd.DataFrame({
                "timestamp_sec": timestamps,
                "decibel_db": [-25.0] * len(timestamps),
                "acoustic_state": ["Instructor Lecture"] * len(timestamps),
                "acoustic_engagement_factor": [80.0] * len(timestamps)
            })

        # Group visual observations by 1-second intervals
        vis_by_sec = {}
        for r in visual_records:
            sec = int(round(r.get("timestamp_sec", 0.0)))
            if sec not in vis_by_sec:
                vis_by_sec[sec] = []
            vis_by_sec[sec].append(r.get("focus_score", 70.0))

        fused_rows = []
        for _, a_row in audio_df.iterrows():
            t_sec = a_row["timestamp_sec"]
            bucket_sec = int(round(t_sec))

            # Visual aggregate for this second
            if bucket_sec in vis_by_sec and len(vis_by_sec[bucket_sec]) > 0:
                vis_score = float(np.mean(vis_by_sec[bucket_sec]))
                active_pts = len(vis_by_sec[bucket_sec])
            else:
                # Interpolate or default to healthy baseline
                vis_score = 72.0
                active_pts = 1

            ac_score = a_row["acoustic_engagement_factor"]
            ac_state = a_row["acoustic_state"]
            ac_db = a_row["decibel_db"]

            fused_score, zone, synergy = self.compute_instantaneous_engagement(
                vis_score, ac_score, ac_state, active_pts
            )

            # Head pose alignment proxy classification
            if vis_score >= 75.0:
                alignment = "Direct Forward Alignment"
            elif vis_score >= 55.0:
                alignment = "Slight Lateral Gaze"
            else:
                alignment = "Slumping / Distracted Head Angle"

            fused_rows.append({
                "timestamp_sec": round(t_sec, 1),
                "visual_attentiveness": round(vis_score, 1),
                "head_pose_alignment": alignment,
                "acoustic_state": ac_state,
                "acoustic_decibel_db": ac_db,
                "acoustic_energy_factor": round(ac_score, 1),
                "synergy_delta": synergy,
                "fused_engagement_index": fused_score,
                "engagement_zone": zone,
                "active_participants": active_pts
            })

        matrix_df = pd.DataFrame(fused_rows)
        # Apply gentle 3-step moving average for smooth display
        if len(matrix_df) >= 3:
            matrix_df["engagement_smooth"] = matrix_df["fused_engagement_index"].rolling(window=3, min_periods=1, center=True).mean().round(1)
        else:
            matrix_df["engagement_smooth"] = matrix_df["fused_engagement_index"]

        return matrix_df

    def detect_peaks_and_dips(self, telemetry_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identifies pedagogical peaks (high collaborative / focused moments)
        and dips (attention drops requiring intervention).
        """
        if telemetry_df.empty or "fused_engagement_index" not in telemetry_df.columns:
            return {"peaks": [], "dips": [], "summary": {}}

        scores = telemetry_df["fused_engagement_index"].values
        times = telemetry_df["timestamp_sec"].values

        peaks = []
        dips = []

        # Local extremum detection with window threshold
        for i in range(1, len(scores) - 1):
            t = times[i]
            s = scores[i]
            if s > scores[i - 1] and s > scores[i + 1] and s >= self.high_threshold:
                peaks.append({
                    "timestamp_sec": round(t, 1),
                    "engagement": round(s, 1),
                    "acoustic_state": telemetry_df.iloc[i].get("acoustic_state", "N/A"),
                    "type": "Attention Peak"
                })
            elif s < scores[i - 1] and s < scores[i + 1] and s <= self.lull_threshold:
                dips.append({
                    "timestamp_sec": round(t, 1),
                    "engagement": round(s, 1),
                    "acoustic_state": telemetry_df.iloc[i].get("acoustic_state", "N/A"),
                    "type": "Attention Lull"
                })

        mean_engagement = float(np.mean(scores))
        peak_score = float(np.max(scores))
        min_score = float(np.min(scores))
        time_above_70 = float(np.sum(scores >= self.high_threshold) / len(scores) * 100.0)
        time_below_45 = float(np.sum(scores < self.lull_threshold) / len(scores) * 100.0)

        return {
            "peaks": peaks,
            "dips": dips,
            "summary": {
                "mean_engagement": round(mean_engagement, 1),
                "peak_engagement": round(peak_score, 1),
                "min_engagement": round(min_score, 1),
                "time_in_high_zone_pct": round(time_above_70, 1),
                "time_in_lull_zone_pct": round(time_below_45, 1)
            }
        }
