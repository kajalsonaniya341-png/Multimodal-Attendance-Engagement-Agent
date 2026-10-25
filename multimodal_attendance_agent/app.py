"""
Direct Launcher for Multimodal Attendance & Engagement Agent
Skill Wallet Capstone Project 5 | Platform ID: 6ac775fb32e2e6be34c17468
"""

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent / "app"
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from streamlit_app import *
