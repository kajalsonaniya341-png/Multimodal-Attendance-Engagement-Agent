import html
import os
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from app.analysis import analyze_video

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"
REPORT_DIR = ROOT / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MAX_BYTES = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

app = FastAPI(
    title="Multimodal Attendance & Engagement Agent",
    description="Local-first classroom video analytics prototype",
    version="1.0.0"
)

# In-memory results are cleared when the server restarts.
RESULTS = {}


def page(content: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Classroom Intelligence</title>
<style>
:root {{ --navy:#12213a; --blue:#4361ee; --light:#f4f7fc; --muted:#667085; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--navy); background:var(--light); }}
header {{ background:linear-gradient(120deg,#172554,#3156c9); color:white; padding:28px max(20px,calc((100% - 1080px)/2)); }}
header .eyebrow {{ font-size:12px; letter-spacing:2px; text-transform:uppercase; opacity:.8; }}
header h1 {{ margin:8px 0; font-size:clamp(25px,4vw,38px); }}
header p {{ margin:0; opacity:.86; }}
main {{ max-width:1080px; margin:28px auto; padding:0 18px 40px; }}
.card {{ background:white; border:1px solid #e4e9f2; border-radius:16px; padding:22px; box-shadow:0 8px 28px #14264b0a; margin-bottom:18px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }}
.metric {{ background:#f8faff; border:1px solid #e8edf8; border-radius:12px; padding:18px; }}
.metric .label {{ color:var(--muted); font-size:13px; }}
.metric .value {{ font-size:26px; font-weight:750; margin-top:8px; }}
h2 {{ margin-top:0; font-size:21px; }}
p {{ line-height:1.55; }}
.muted {{ color:var(--muted); font-size:13px; }}
button,.button {{ border:0; border-radius:9px; padding:12px 17px; background:var(--blue); color:white; font-weight:650; cursor:pointer; text-decoration:none; display:inline-block; }}
button:disabled {{ opacity:.55; cursor:wait; }}
input[type=file] {{ width:100%; padding:14px; border:1px dashed #aab7d4; border-radius:10px; background:#fafcff; margin:12px 0; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ text-align:left; border-bottom:1px solid #edf0f5; padding:11px 8px; }}
th {{ color:#526078; }}
.table-wrap {{ overflow-x:auto; }}
.notice {{ background:#fff8e7; border:1px solid #f4dfaa; padding:13px; border-radius:10px; color:#6b4d0d; }}
.success {{ color:#087443; }}
footer {{ text-align:center; padding:20px; color:var(--muted); font-size:12px; }}
a {{ color:#3156c9; }}
@media(max-width:600px) {{ header {{ padding:24px 18px; }} .card {{ padding:17px; }} }}
</style>
</head>
<body>
<header>
<div class="eyebrow">AI • Computer Vision • Audio Analytics</div>
<h1>Classroom Intelligence</h1>
<p>Multimodal Attendance &amp; Engagement Agent</p>
</header>
<main>{content}</main>
<footer>Local-first student prototype · Use only authorized, consented recordings</footer>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    content = """
    <section class="card">
      <h2>Analyze a classroom recording</h2>
      <p>Upload a short video to estimate anonymous person tracks, observed presence,
      movement proxies and optional audio activity.</p>
      <form action="/analyze" method="post" enctype="multipart/form-data" id="uploadForm">
        <label for="video"><b>Choose video file</b></label>
        <input id="video" name="video" type="file" accept=".mp4,.avi,.mov,.mkv" required>
        <button type="submit" id="submitBtn">Analyze video</button>
        <p class="muted">Maximum 100 MB. Start with a 10–30 second clip. Processing may take time on CPU.</p>
      </form>
    </section>
    <section class="grid">
      <div class="metric"><div class="label">Computer vision</div><div class="value">YOLO</div><p class="muted">Person detection and tracking</p></div>
      <div class="metric"><div class="label">Video processing</div><div class="value">OpenCV</div><p class="muted">Frames, duration and movement</p></div>
      <div class="metric"><div class="label">Audio signal</div><div class="value">RMS</div><p class="muted">Optional sound-level timeline</p></div>
    </section>
    <section class="card">
      <h2>How to use</h2>
      <ol>
        <li>Use a video you have permission to analyze.</li>
        <li>Click Choose video and select an MP4/AVI/MOV/MKV file.</li>
        <li>Click Analyze video and wait for the result page.</li>
        <li>Download the CSV attendance estimate if needed.</li>
      </ol>
      <div class="notice"><b>Important:</b> anonymous tracking IDs are not student identities.
      Movement, head direction and audio level do not prove attention, emotion or learning.
      Do not use these estimates as the sole basis for penalties.</div>
    </section>
    <script>
    document.getElementById('uploadForm').addEventListener('submit', function() {
      const b = document.getElementById('submitBtn');
      b.disabled = true; b.textContent = 'Analyzing… please wait';
    });
    </script>
    """
    return page(content)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(video: UploadFile = File(...)):
    original_name = Path(video.filename or "classroom_video").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        await video.close()
        raise HTTPException(status_code=400, detail="Unsupported file type. Use MP4, AVI, MOV or MKV.")

    job_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{job_id}{extension}"
    total = 0

    try:
        with path.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="File is larger than 100 MB.")
                out.write(chunk)

        if total == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        try:
            result = analyze_video(str(path), REPORT_DIR / job_id)
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}. Run pip install -r requirements.txt.")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Analysis failed: {str(exc)}")

        result["job_id"] = job_id
        result["source_filename"] = original_name
        RESULTS[job_id] = result

        rows = result["attendance_rows"]
        table_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(r['anonymous_track_id']))}</td>"
            f"<td>{r['first_seen_seconds']:.1f}s</td>"
            f"<td>{r['last_seen_seconds']:.1f}s</td>"
            f"<td>{r['estimated_observed_seconds']:.1f}s</td>"
            f"<td>{r['detections']}</td>"
            f"<td>{r['average_center_shift_pixels_per_sample']:.1f}</td>"
            "</tr>" for r in rows
        )

        audio = result["audio"]
        if audio.get("available"):
            audio_html = (
                f"<p><b>Mean audio RMS:</b> {audio['mean_rms']}</p>"
                f"<p class='muted'>{html.escape(audio['message'])}</p>"
                "<div class='table-wrap'><table><thead><tr><th>Interval start</th><th>RMS level</th></tr></thead><tbody>"
                + "".join(f"<tr><td>{x['interval_start_seconds']}s</td><td>{x['rms_level']}</td></tr>" for x in audio["intervals"][:100])
                + "</tbody></table></div>"
            )
        else:
            audio_html = f"<p>{html.escape(audio.get('message', 'Audio analysis unavailable.'))}</p>"

        suggestions = make_suggestions(result)
        suggestion_html = "".join(f"<li>{html.escape(s)}</li>" for s in suggestions)

        content = f"""
        <section class="card">
          <p class="success"><b>Analysis completed</b></p>
          <h2>{html.escape(original_name)}</h2>
          <div class="grid">
            <div class="metric"><div class="label">Video duration</div><div class="value">{result['duration_display']}</div></div>
            <div class="metric"><div class="label">Anonymous tracks</div><div class="value">{result['unique_track_count']}</div></div>
            <div class="metric"><div class="label">Sampled frames</div><div class="value">{result['sampled_frames']}</div></div>
            <div class="metric"><div class="label">Resolution</div><div class="value">{result['width']}×{result['height']}</div></div>
          </div>
          <p class="muted">Frame rate: {result['fps']} FPS · Person detections: {result['person_detections']}</p>
          <a class="button" href="/reports/{job_id}">Download attendance CSV</a>
          <a href="/api/results/{job_id}" style="margin-left:12px">View JSON result</a>
        </section>
        <section class="card">
          <h2>Estimated presence and movement</h2>
          <p class="muted">These are anonymous model tracks, not verified people. Tracks may split or switch.</p>
          <div class="table-wrap"><table>
          <thead><tr><th>Track</th><th>First seen</th><th>Last seen</th><th>Observed time*</th><th>Detections</th><th>Center shift</th></tr></thead>
          <tbody>{table_rows}</tbody></table></div>
          <p class="muted">*Observed time is an estimate based on sampled detections and tracking continuity, not official attendance.</p>
        </section>
        <section class="card">
          <h2>Audio activity</h2>{audio_html}
        </section>
        <section class="card">
          <h2>Teaching suggestions</h2>
          <ul>{suggestion_html}</ul>
          <p class="muted">Suggestions are rule-based prompts for human review, not judgments about student attention.</p>
        </section>
        <section class="card"><h2>Responsible use</h2><p>{html.escape(result['notice'])}</p><a href="/">Analyze another video</a></section>
        """
        return page(content)
    finally:
        await video.close()
        if path.exists():
            path.unlink()


def make_suggestions(result):
    suggestions = []
    rows = result.get("attendance_rows", [])
    if len(rows) > 0:
        suggestions.append(
            f"The system detected {len(rows)} anonymous person tracks. Review the video if an accurate roster is needed."
        )
    audio = result.get("audio", {})
    if audio.get("available") and audio.get("mean_rms", 0) < 500:
        suggestions.append(
            "Average audio energy was relatively low. Check the recording volume and session context before interpreting this result."
        )
    elif audio.get("available"):
        suggestions.append(
            "Review intervals with changing sound levels to determine whether they correspond to discussion, teacher speech, or background noise."
        )
    suggestions.append(
        "Consider a brief question-and-answer or feedback activity, then compare the next session's results and learner feedback."
    )
    return suggestions


@app.get("/reports/{job_id}")
def download_report(job_id: str):
    result = RESULTS.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Report not found. Run the analysis again.")
    csv_path = REPORT_DIR / job_id / "attendance_report.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV report file not found.")
    return FileResponse(csv_path, media_type="text/csv", filename="attendance_report.csv")


@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    result = RESULTS.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found. Run the analysis again.")
    return JSONResponse(result)


@app.get("/health")
def health():
    return {"status": "running", "project": "Multimodal Attendance & Engagement Agent"}
