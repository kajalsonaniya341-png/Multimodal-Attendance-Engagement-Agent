"""
Synthetic Classroom Telemetry & Demonstration Generator
Privacy-First Design: Generates synthetic test feeds for 1-click demonstration & automated tests.
Platform ID: 6ac775fb32e2e6be34c17468
"""

import math
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Any


class SyntheticClassroomGenerator:
    """
    Generates realistic synthetic classroom sessions to enable instant 1-click
    demonstrations of presence tracking, acoustic classification, and multimodal fusion.
    """

    def __init__(self, num_students: int = 6, duration_sec: int = 90):
        self.num_students = num_students
        self.duration_sec = duration_sec

    def generate_synthetic_session_telemetry(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """
        Synthesizes a full classroom session with realistic student behaviors:
        - Students 1 to 5: Sustained presence (85% to 98% of duration), varying attentiveness.
        - Student 6: Transient hallway pass-through (appears only for ~15% duration, fails >70% threshold).
        - Acoustic stages: Lecture -> Discussion -> High Distraction -> Silence -> Lecture.
        """
        timestamps = np.arange(0, self.duration_sec, 1.0)
        telemetry_rows = []
        visual_records = []

        # Attendance roster metadata
        students_meta = [
            {"id": "STUDENT_001", "base_focus": 88, "presence_span": (0, 90), "role": "Highly Attentive Front Row"},
            {"id": "STUDENT_002", "base_focus": 82, "presence_span": (2, 88), "role": "Active Participant"},
            {"id": "STUDENT_003", "base_focus": 74, "presence_span": (0, 85), "role": "Moderate Attentiveness"},
            {"id": "STUDENT_004", "base_focus": 64, "presence_span": (5, 90), "role": "Occasional Slumping/Phone"},
            {"id": "STUDENT_005", "base_focus": 78, "presence_span": (1, 89), "role": "Engaged Peer Collaborator"},
            {"id": "STUDENT_006", "base_focus": 45, "presence_span": (35, 48), "role": "Hallway Pass-Through / Transient"}
        ]

        # Attendance summary table calculation
        attendance_rows = []
        for s in students_meta:
            start_t, end_t = s["presence_span"]
            duration = end_t - start_t
            rate = round((duration / self.duration_sec) * 100.0, 1)
            status = "Present" if rate >= 70.0 else "Absent (Transient / <70%)"
            attendance_rows.append({
                "Student ID": s["id"],
                "First Seen (s)": float(start_t),
                "Last Seen (s)": float(end_t),
                "Total Duration (s)": float(duration),
                "Presence Rate (%)": rate,
                "Status": status,
                "Attentiveness Avg (%)": float(s["base_focus"]),
                "Role / Profile": s["role"]
            })
        attendance_df = pd.DataFrame(attendance_rows)

        # Generate second-by-second multimodal fusion stream
        for t in timestamps:
            # 1. Determine acoustic state based on lesson phase
            if 0 <= t < 22:
                ac_state = "Instructor Lecture"
                ac_energy = 80.0
                ac_db = -24.0 + 3.0 * math.sin(t * 0.5)
            elif 22 <= t < 50:
                ac_state = "Student Discussion / Verbal Interaction"
                ac_energy = 92.0
                ac_db = -18.0 + 5.0 * math.cos(t * 0.8)
            elif 50 <= t < 65:
                ac_state = "High Ambient Classroom Distraction"
                ac_energy = 25.0
                ac_db = -11.0 + 4.0 * math.sin(t * 1.5)
            elif 65 <= t < 78:
                ac_state = "Silence"
                ac_energy = 65.0
                ac_db = -46.0 + 2.0 * math.cos(t * 0.2)
            else:
                ac_state = "Instructor Lecture"
                ac_energy = 82.0
                ac_db = -22.0 + 2.5 * math.sin(t * 0.4)

            # 2. Determine collective visual score from active students at time t
            active_foci = []
            for s in students_meta:
                start_t, end_t = s["presence_span"]
                if start_t <= t <= end_t:
                    # Add temporal fluctuation & situational fatigue
                    fatigue = max(0.0, (t - 40) * 0.15) if t > 40 else 0.0
                    noise_penalty = 12.0 if ac_state == "High Ambient Classroom Distraction" else 0.0
                    discussion_boost = 6.0 if ac_state == "Student Discussion / Verbal Interaction" else 0.0
                    fluctuation = 6.0 * math.sin(t * 0.3 + int(s["id"][-1]))
                    instant_focus = float(np.clip(s["base_focus"] - fatigue - noise_penalty + discussion_boost + fluctuation, 15.0, 98.0))
                    active_foci.append(instant_focus)

                    visual_records.append({
                        "timestamp_sec": t,
                        "student_id": s["id"],
                        "focus_score": instant_focus
                    })

            vis_avg = float(np.mean(active_foci)) if active_foci else 70.0

            # 3. Multimodal fusion
            raw = (0.60 * vis_avg) + (0.40 * ac_energy)
            synergy = 0.0
            if ac_state == "Student Discussion / Verbal Interaction" and vis_avg >= 65:
                synergy = 6.0
            elif ac_state == "High Ambient Classroom Distraction":
                synergy = -14.0
            elif ac_state == "Silence" and vis_avg < 50:
                synergy = -10.0

            fused_score = float(np.clip(raw + synergy, 10.0, 100.0))

            if fused_score >= 70.0:
                zone = "High Engagement"
            elif fused_score >= 45.0:
                zone = "Moderate / Passive Focus"
            else:
                zone = "Attention Deficit / Lull"

            head_alignment = "Direct Forward Alignment" if vis_avg >= 75 else "Slight Lateral Gaze" if vis_avg >= 55 else "Slumping / Distracted Head Angle"

            telemetry_rows.append({
                "timestamp_sec": round(t, 1),
                "visual_attentiveness": round(vis_avg, 1),
                "head_pose_alignment": head_alignment,
                "acoustic_state": ac_state,
                "acoustic_decibel_db": round(ac_db, 1),
                "acoustic_energy_factor": round(ac_energy, 1),
                "synergy_delta": round(synergy, 1),
                "fused_engagement_index": round(fused_score, 1),
                "engagement_zone": zone,
                "active_participants": len(active_foci)
            })

        telemetry_df = pd.DataFrame(telemetry_rows)
        telemetry_df["engagement_smooth"] = telemetry_df["fused_engagement_index"].rolling(window=3, min_periods=1, center=True).mean().round(1)

        return attendance_df, telemetry_df, visual_records

    def render_synthetic_classroom_frame(self, t_sec: float) -> np.ndarray:
        """
        Draws a realistic stylized simulated classroom video frame with student desks,
        head pose orientation vectors, and ethical privacy telemetry bounding boxes.
        """
        w, h = 760, 440
        frame = np.full((h, w, 3), (242, 246, 248), dtype=np.uint8)

        # Classroom board / lectern in background
        cv2.rectangle(frame, (80, 20), (680, 80), (45, 60, 75), -1)
        cv2.putText(
            frame, "PRIVACY-PRESERVING CLASSROOM TELEMETRY (EDGE INFERENCE)",
            (100, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 245, 255), 1, cv2.LINE_AA
        )

        # Desks and student avatars (2 rows of 3 desks)
        desk_positions = [
            (120, 160), (330, 160), (540, 160),
            (120, 280), (330, 280), (540, 280)
        ]

        for i, (dx, dy) in enumerate(desk_positions):
            sid = f"STUDENT_{i+1:03d}"
            # Desk
            cv2.rectangle(frame, (dx - 45, dy + 50), (dx + 85, dy + 85), (205, 215, 225), -1)
            cv2.rectangle(frame, (dx - 45, dy + 50), (dx + 85, dy + 85), (170, 185, 200), 2)

            # Check transient student #6
            if i == 5 and (t_sec < 35 or t_sec > 48):
                # Hallway empty desk
                cv2.putText(frame, "Empty Desk", (dx - 20, dy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 170), 1)
                continue

            # Student Avatar
            head_color = (190, 220, 190) if i != 5 else (180, 190, 230)
            cv2.circle(frame, (dx + 20, dy + 15), 25, head_color, -1)
            cv2.circle(frame, (dx + 20, dy + 15), 25, (80, 100, 90), 2)

            # Head pose simulation: vector pointing to board
            yaw_tilt = int(12 * math.sin(t_sec * 0.4 + i))
            cv2.line(frame, (dx + 20, dy + 15), (dx + 20 + yaw_tilt, dy - 8), (40, 60, 180), 2)

            # Ethical Telemetry Bounding Box
            box_x, box_y, box_w, box_h = dx - 15, dy - 18, 70, 65
            border_color = (0, 180, 60) if i != 5 else (0, 140, 240)
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), border_color, 2)

            # Metadata label
            badge_text = f"{sid}"
            cv2.rectangle(frame, (box_x, box_y - 18), (box_x + 65, box_y), border_color, -1)
            cv2.putText(frame, badge_text, (box_x + 4, box_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Session watermark
        time_text = f"Classroom Time: {int(t_sec // 60):02d}:{int(t_sec % 60):02d} | 100% Offline Edge"
        cv2.putText(frame, time_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 90, 100), 1)

        return frame
