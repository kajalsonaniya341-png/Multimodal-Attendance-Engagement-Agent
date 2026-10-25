# Testing checklist

1. Start server and open `/health`; expected JSON includes `"status": "running"`.
2. Upload a short valid video with visible people.
3. Confirm duration, FPS and resolution are shown.
4. Confirm anonymous person tracks appear and CSV downloads.
5. If FFmpeg is installed, confirm audio RMS intervals are shown.
6. Try an unsupported file extension and confirm it is rejected.
7. Try a video without visible people and confirm the app shows an analysis error.
8. Test with consented footage only.
