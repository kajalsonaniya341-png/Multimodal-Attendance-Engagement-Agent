import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

MODEL_NAME = os.getenv("YOLO_MODEL", "yolo11n.pt")


def analyze_video(video_path: str, report_dir: Path, progress_callback=None):
    """Analyze a video using person detection/tracking and simple observable proxies."""
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video. Try a different MP4/AVI/MOV/MKV file.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError("Video duration or frame rate could not be read.")

    duration = frame_count / fps
    # Sample at roughly 2 frames/second to keep CPU processing manageable.
    sample_every = max(1, int(round(fps / 2)))
    model = YOLO(MODEL_NAME)

    tracks = defaultdict(lambda: {"first": None, "last": None, "seen_seconds": 0.0,
                                  "centers": [], "detections": 0})
    total_people_detections = 0
    sampled = 0
    frame_index = 0
    last_progress = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            timestamp = frame_index / fps
            # COCO class 0 is person. persist=True enables tracking between calls.
            result = model.track(frame, persist=True, classes=[0], verbose=False,
                                 conf=0.35, tracker="bytetrack.yaml")[0]
            sampled += 1
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy)) + (sampled * 10000)
                for box, track_id in zip(xyxy, ids):
                    x1, y1, x2, y2 = box.tolist()
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    item = tracks[int(track_id)]
                    if item["first"] is None:
                        item["first"] = timestamp
                    if item["last"] is not None:
                        gap = timestamp - item["last"]
                        # Count only reasonable gaps; occlusion can otherwise inflate presence.
                        if 0 < gap <= 2.5:
                            item["seen_seconds"] += gap
                    item["last"] = timestamp
                    item["centers"].append((timestamp, center[0], center[1]))
                    item["detections"] += 1
                    total_people_detections += 1

            percent = int(min(95, (frame_index / max(1, frame_count)) * 95))
            if progress_callback and percent != last_progress:
                progress_callback(percent)
                last_progress = percent
        frame_index += 1

    cap.release()

    if not tracks:
        raise ValueError("No people were detected. Try a clearer video with people visible.")

    rows = []
    for tid, item in tracks.items():
        first = item["first"] or 0.0
        last = item["last"] or first
        observed = min(duration, max(0.0, item["seen_seconds"]))
        # Include a small sample interval for a track seen only once.
        if item["detections"] == 1:
            observed = min(duration, 0.5)
        centers = item["centers"]
        movement = 0.0
        if len(centers) >= 2:
            distances = []
            for a, b in zip(centers, centers[1:]):
                distances.append(float(np.hypot(b[1] - a[1], b[2] - a[2])))
            movement = float(np.mean(distances))
        rows.append({
            "anonymous_track_id": f"Person-{tid}",
            "first_seen_seconds": round(first, 2),
            "last_seen_seconds": round(last, 2),
            "estimated_observed_seconds": round(observed, 2),
            "detections": int(item["detections"]),
            "average_center_shift_pixels_per_sample": round(movement, 2),
        })

    attendance_df = pd.DataFrame(rows).sort_values("anonymous_track_id")
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "attendance_report.csv"
    attendance_df.to_csv(csv_path, index=False)

    audio = analyze_audio(video_path)
    if progress_callback:
        progress_callback(100)

    # Summary metrics are descriptive, not a psychological engagement score.
    summary = {
        "duration_seconds": round(duration, 2),
        "duration_display": f"{int(duration // 60)} min {int(duration % 60)} sec",
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "sampled_frames": sampled,
        "unique_track_count": len(tracks),
        "person_detections": total_people_detections,
        "attendance_rows": rows,
        "audio": audio,
        "report_filename": csv_path.name,
        "notice": "Anonymous tracks and activity proxies are estimates, not verified identities or proof of attention/learning."
    }
    return summary


def analyze_audio(video_path: str):
    """Optional FFmpeg extraction + RMS activity summary. Gracefully skips if FFmpeg missing."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {
            "available": False,
            "message": "Audio analysis skipped: FFmpeg is not installed or not on PATH."
        }

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = str(Path(temp_dir) / "audio.wav")
        command = [
            ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1",
            "-ar", "16000", "-f", "wav", wav_path
        ]
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {"available": False, "message": "Audio track could not be extracted."}

        try:
            import wave
            with wave.open(wav_path, "rb") as wav:
                rate = wav.getframerate()
                width = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())
            if width != 2:
                return {"available": False, "message": "Unsupported extracted audio sample format."}
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            if len(samples) == 0:
                return {"available": False, "message": "No audio samples found."}
            chunk_size = max(1, rate * 5)
            levels = []
            for start in range(0, len(samples), chunk_size):
                chunk = samples[start:start + chunk_size]
                rms = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
                levels.append({
                    "interval_start_seconds": round(start / rate, 1),
                    "rms_level": round(rms, 2)
                })
            return {
                "available": True,
                "message": "Audio RMS measures sound level; it does not prove speech or engagement.",
                "intervals": levels,
                "mean_rms": round(float(np.mean([x["rms_level"] for x in levels])), 2)
            }
        except Exception:
            return {"available": False, "message": "Audio analysis failed for this file."}
