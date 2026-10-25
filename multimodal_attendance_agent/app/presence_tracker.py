"""
Presence Tracker & Face Telemetry Module
EPIC 2: Face Tracking & Sustained Presence Attendance
Privacy-First Design: Offline Edge Processing & Ethical Anonymized Telemetry
Platform ID: 6ac775fb32e2e6be34c17468
"""

import time
import math
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import cv2

# Approximate 3D model points for Head Pose Estimation (PnP)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)


class FaceTrack:
    """Represents an anonymized participant tracking state across video frames."""

    def __init__(self, track_id: int, initial_box: Tuple[int, int, int, int], timestamp: float):
        self.track_id = track_id
        self.anonymized_id = f"STUDENT_{track_id:03d}"
        self.box = initial_box  # (x, y, w, h)
        self.centroid = (initial_box[0] + initial_box[2] // 2, initial_box[1] + initial_box[3] // 2)
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.detection_count = 1
        self.lost_frames = 0
        self.observed_seconds = 0.0
        self.head_pose_history: List[Dict[str, float]] = []
        self.attentiveness_history: List[float] = []

    def update(self, box: Tuple[int, int, int, int], timestamp: float, pose_data: Optional[Dict[str, float]] = None):
        delta = timestamp - self.last_seen
        # Count continuous observation if gap is reasonable (< 3.0s)
        if 0 < delta <= 3.0:
            self.observed_seconds += delta
        elif delta > 3.0:
            self.observed_seconds += 0.5  # re-detection burst

        self.box = box
        self.centroid = (box[0] + box[2] // 2, box[1] + box[3] // 2)
        self.last_seen = timestamp
        self.detection_count += 1
        self.lost_frames = 0

        if pose_data:
            self.head_pose_history.append(pose_data)
            self.attentiveness_history.append(pose_data.get("focus_score", 70.0))

    def mark_lost(self):
        self.lost_frames += 1


class PresenceTracker:
    """
    Offline Face Detection, Tracking ID Assignment, and Sustained Presence Attendance.
    Implements >70% sustained duration threshold to prevent hallway pass-through false marks.
    """

    def __init__(self, sustained_threshold: float = 0.70, max_disappeared_frames: int = 30):
        self.sustained_threshold = sustained_threshold  # 70% default
        self.max_disappeared_frames = max_disappeared_frames
        self.next_track_id = 1
        self.active_tracks: Dict[int, FaceTrack] = {}
        self.completed_tracks: Dict[int, FaceTrack] = {}
        self.session_start: Optional[float] = None
        self.session_end: Optional[float] = None

        # Initialize MediaPipe or OpenCV fallback
        self.mp_face_mesh = None
        self.face_mesh = None
        self._init_detector()

    def _init_detector(self):
        """Initialize face detector with graceful fallback."""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=20,
                refine_landmarks=True,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45
            )
            self.detector_type = "MediaPipe Face Mesh"
        except Exception:
            self.detector_type = "OpenCV Cascade Fallback"
            cascade_dir = getattr(cv2.data, "haarcascades", "")
            cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml") if cascade_dir else ""
            if cascade_path and os.path.exists(cascade_path):
                self.cascade = cv2.CascadeClassifier(cascade_path)
            else:
                self.cascade = None

    def _estimate_head_pose(self, landmarks_2d: np.ndarray, img_w: int, img_h: int) -> Dict[str, float]:
        """
        Calculates Euler angles (Pitch, Yaw, Roll) using OpenCV solvePnP.
        Determines if head pose is aligned forward (attentive) vs looking away/slumping.
        """
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            MODEL_POINTS_3D, landmarks_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "focus_score": 75.0, "state": "Forward"}

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        # Compute Euler angles
        sy = math.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
            yaw = math.atan2(-rot_mat[2, 0], sy)
            roll = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            pitch = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
            yaw = math.atan2(-rot_mat[2, 0], sy)
            roll = 0.0

        pitch_deg = round(math.degrees(pitch), 1)
        yaw_deg = round(math.degrees(yaw), 1)
        roll_deg = round(math.degrees(roll), 1)

        # Attentiveness scoring logic:
        # Ideal forward attention: |yaw| <= 20 deg, -15 deg <= pitch <= 20 deg
        abs_yaw = abs(yaw_deg)
        # Pitch penalty: looking down at desk/phone (pitch < -20) or looking up/away
        yaw_penalty = max(0.0, (abs_yaw - 15.0) * 1.8)
        pitch_penalty = max(0.0, (abs(pitch_deg) - 15.0) * 1.5) if pitch_deg > 0 else max(0.0, (abs(pitch_deg) - 12.0) * 2.2)

        focus_score = max(5.0, min(100.0, 100.0 - yaw_penalty - pitch_penalty))

        # Classify state
        if abs_yaw > 35.0:
            state = "Looking Sideways / Turning"
        elif pitch_deg < -22.0:
            state = "Downward / Slumping"
        elif pitch_deg > 25.0:
            state = "Upward / Distracted"
        else:
            state = "Attentive Forward"

        return {
            "yaw": yaw_deg,
            "pitch": pitch_deg,
            "roll": roll_deg,
            "focus_score": round(focus_score, 1),
            "state": state
        }

    def detect_and_track_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        apply_privacy_blur: bool = False
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single video/webcam frame:
        1. Detect faces.
        2. Assign/update tracking IDs.
        3. Estimate head pose & attentiveness.
        4. Render ethical privacy bounding boxes and telemetry.
        """
        if self.session_start is None:
            self.session_start = timestamp
        self.session_end = timestamp

        h, w = frame.shape[:2]
        detected_boxes: List[Tuple[int, int, int, int]] = []
        detected_poses: List[Optional[Dict[str, float]]] = []

        # 1. Detection
        if self.detector_type == "MediaPipe Face Mesh" and self.face_mesh is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Key landmarks for 2D-3D correspondence:
                    # Nose tip (1), Chin (152), Left Eye corner (33), Right Eye corner (263), Left Mouth (61), Right Mouth (291)
                    lm_indices = [1, 152, 33, 263, 61, 291]
                    pts_2d = []
                    xs, ys = [], []
                    for lm in face_landmarks.landmark:
                        xs.append(int(lm.x * w))
                        ys.append(int(lm.y * h))
                    min_x, max_x = max(0, min(xs)), min(w, max(xs))
                    min_y, max_y = max(0, min(ys)), min(h, max(ys))
                    box_w = max_x - min_x
                    box_h = max_y - min_y

                    if box_w > 20 and box_h > 20:
                        detected_boxes.append((min_x, min_y, box_w, box_h))
                        for idx in lm_indices:
                            lm = face_landmarks.landmark[idx]
                            pts_2d.append((float(lm.x * w), float(lm.y * h)))
                        pose = self._estimate_head_pose(np.array(pts_2d, dtype=np.float64), w, h)
                        detected_poses.append(pose)
        else:
            # Fallback OpenCV Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, bw, bh) in faces:
                detected_boxes.append((int(x), int(y), int(bw), int(bh)))
                # Approximate proxy pose for cascade fallback
                detected_poses.append({
                    "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "focus_score": 80.0, "state": "Attentive Forward"
                })

        # 2. Tracking ID assignment (Centroid + IoU distance)
        matched_tracks = set()
        current_frame_data = []

        for box, pose in zip(detected_boxes, detected_poses):
            cx = box[0] + box[2] // 2
            cy = box[1] + box[3] // 2

            best_track_id = None
            min_dist = float("inf")

            for tid, track in self.active_tracks.items():
                if tid in matched_tracks:
                    continue
                tcx, tcy = track.centroid
                dist = math.hypot(cx - tcx, cy - tcy)
                # Allowable displacement threshold scaled by box size
                max_allowed_dist = max(box[2], box[3]) * 1.5
                if dist < max_allowed_dist and dist < min_dist:
                    min_dist = dist
                    best_track_id = tid

            if best_track_id is not None:
                track = self.active_tracks[best_track_id]
                track.update(box, timestamp, pose)
                matched_tracks.add(best_track_id)
            else:
                # Create new track
                new_track = FaceTrack(self.next_track_id, box, timestamp)
                new_track.update(box, timestamp, pose)
                self.active_tracks[self.next_track_id] = new_track
                matched_tracks.add(self.next_track_id)
                self.next_track_id += 1

        # Handle lost tracks
        lost_to_archive = []
        for tid, track in self.active_tracks.items():
            if tid not in matched_tracks:
                track.mark_lost()
                if track.lost_frames > self.max_disappeared_frames:
                    lost_to_archive.append(tid)

        for tid in lost_to_archive:
            self.completed_tracks[tid] = self.active_tracks.pop(tid)

        # 3. Privacy Rendering & Visual Telemetry Overlay
        output_frame = frame.copy()
        total_session_elapsed = max(1.0, timestamp - (self.session_start or timestamp))

        for tid in matched_tracks:
            track = self.active_tracks[tid]
            x, y, bw, bh = track.box
            presence_ratio = min(1.0, track.observed_seconds / total_session_elapsed)
            is_sustained = presence_ratio >= self.sustained_threshold

            # Privacy Blur: blur face region if requested
            if apply_privacy_blur:
                face_roi = output_frame[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
                if face_roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
                    output_frame[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)] = blurred_roi

            # Color coding: Green for sustained presence, Orange/Yellow for evaluating
            color = (0, 200, 80) if is_sustained else (0, 165, 255)
            cv2.rectangle(output_frame, (x, y), (x + bw, y + bh), color, 2)

            # Metadata Badge (Ethical anonymized telemetry only, no personal identifiers)
            latest_focus = track.attentiveness_history[-1] if track.attentiveness_history else 75.0
            badge_label = f"{track.anonymized_id} | Pres: {int(presence_ratio * 100)}% | Focus: {int(latest_focus)}%"
            cv2.rectangle(output_frame, (x, max(0, y - 24)), (x + len(badge_label) * 8 + 10, max(0, y)), color, -1)
            cv2.putText(
                output_frame, badge_label, (x + 4, max(12, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )

            current_frame_data.append({
                "student_id": track.anonymized_id,
                "track_id": track.track_id,
                "presence_ratio": presence_ratio,
                "is_sustained": is_sustained,
                "focus_score": latest_focus,
                "box": (x, y, bw, bh)
            })

        return output_frame, current_frame_data

    def get_attendance_register(self, total_session_duration: Optional[float] = None) -> pd.DataFrame:
        """
        Computes formal attendance register with Sustained Presence Verification.
        Threshold: participant detected for >70% of total session time to prevent hallway passes.
        """
        all_tracks = {**self.completed_tracks, **self.active_tracks}
        if not all_tracks:
            return pd.DataFrame(columns=[
                "Student ID", "First Seen (s)", "Last Seen (s)",
                "Total Duration (s)", "Presence Rate (%)", "Status", "Attentiveness Avg (%)"
            ])

        if total_session_duration is None or total_session_duration <= 0:
            if self.session_end is not None and self.session_start is not None:
                total_session_duration = max(1.0, self.session_end - self.session_start)
            else:
                total_session_duration = max(1.0, max((t.last_seen for t in all_tracks.values()), default=1.0))

        rows = []
        for tid, track in sorted(all_tracks.items(), key=lambda item: item[1].anonymized_id):
            observed = min(total_session_duration, max(0.5, track.observed_seconds))
            presence_rate = round(min(100.0, (observed / total_session_duration) * 100.0), 1)

            # Core Requirement: Sustained Presence Threshold (>70%)
            threshold_pct = self.sustained_threshold * 100.0
            if presence_rate >= threshold_pct:
                status = "Present"
            else:
                status = "Absent (Transient / <70%)"

            avg_focus = round(float(np.mean(track.attentiveness_history)), 1) if track.attentiveness_history else 70.0

            rows.append({
                "Student ID": track.anonymized_id,
                "First Seen (s)": round(track.first_seen, 1),
                "Last Seen (s)": round(track.last_seen, 1),
                "Total Duration (s)": round(observed, 1),
                "Presence Rate (%)": presence_rate,
                "Status": status,
                "Attentiveness Avg (%)": avg_focus
            })

        df = pd.DataFrame(rows)
        return df

    def reset(self):
        """Reset state for a new session."""
        self.next_track_id = 1
        self.active_tracks.clear()
        self.completed_tracks.clear()
        self.session_start = None
        self.session_end = None
