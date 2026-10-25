"""
Root Launcher for Multimodal Attendance & Engagement Agent
Skill Wallet Capstone Project 5 | Platform ID: 6ac775fb32e2e6be34c17468
"""

import sys
from pathlib import Path

# Ensure inner modules can be imported directly
root_dir = Path(__file__).resolve().parent
inner_app_dir = root_dir / "multimodal_attendance_agent" / "app"
if str(inner_app_dir) not in sys.path:
    sys.path.insert(0, str(inner_app_dir))

# Execute the main Streamlit application
from streamlit_app import *
