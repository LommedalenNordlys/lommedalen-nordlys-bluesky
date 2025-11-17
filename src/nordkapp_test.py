#!/usr/bin/env python3
"""Nordkapp Aurora Test Runner

Purpose: Validate darkness + KP gating and image analysis pipeline on a high-latitude
location (Nordkapp) where aurora occurrences are more frequent. If aurora is detected
and confirmed by AI, post a TEST message to Bluesky using the existing post_to_bluesky
function with a test disclaimer.

Logic:
1. Darkness gate using sun_schedule_nordkapp.json (hour-based fast check)
2. KP / aurora potential gate via shell script check_kp_multi_nordkapp.sh (NOAA + YR.no OR logic)
3. Fetch localized KP proxy via Ovation for inclusion in post (optional)
4. Download Nordkapp webcam image and run color heuristic
5. If heuristic passes, ask AI for confirmation
6. Post test message on positive AI confirmation (test flag set True)

Artifacts:
- Logs appended to kp_data_nordkapp via shell script already
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

# Ensure we can import functions from main
sys.path.append("src")  # allow `import main`
from main import (
    download_image,
    check_for_aurora_colors,
    ask_ai_if_aurora,
    post_to_bluesky,
    fetch_current_kp,
)

SUN_SCHEDULE_FILE = Path("sun_schedule_nordkapp.json")
WEBCAM_FILE = Path("webcam_nordkapp.json")
IMAGE_TEMP_DIR = Path("today")
IMAGE_TEMP_DIR.mkdir(exist_ok=True)

NORDKAPP_LAT = 70.977376
NORDKAPP_LON = 24.6618412
MIN_KP = float(os.getenv("MIN_KP", "1"))

def hour_darkness_gate(schedule_file: Path) -> bool:
    """Fast darkness check using dark_hours in schedule file (same schema as original)."""
    if not schedule_file.exists():
        return True  # fail-open
    try:
        data = json.loads(schedule_file.read_text())
        dh = data.get("dark_hours", {})
        start = dh.get("start_hour")
        end = dh.get("end_hour")
        spans_midnight = dh.get("spans_midnight")
        if start is None or end is None:
            return True
        # Use timezone-aware UTC (Python 3.11+): datetime.now(UTC)
        hour = int(datetime.now(UTC).strftime("%H"))
        if spans_midnight:
            return hour >= start or hour <= end
        else:
            return start <= hour <= end
    except Exception:
        return True

def kp_gate_via_script() -> bool:
    """Invoke the Nordkapp multi-source KP script; return True if potential detected."""
    script = Path("src/check_kp_multi_nordkapp.sh")
    if not script.exists():
        return True  # fail-open
    try:
        # Run with default thresholds (MIN_KP env not directly used in script call here)
        proc = subprocess.run(["bash", str(script), str(int(MIN_KP)), "1.5"], capture_output=True, text=True, timeout=120)
        print(proc.stdout)
        return proc.returncode == 0
    except Exception as e:
        print(f"KP script error: {e}; proceeding fail-open")
        return True

def load_webcam_url() -> Optional[str]:
    if not WEBCAM_FILE.exists():
        return None
    try:
        data = json.loads(WEBCAM_FILE.read_text())
        cams = data.get("webcams", [])
        if not cams:
            return None
        urls = cams[0].get("urls", [])
        return urls[0] if urls else None
    except Exception:
        return None

def main():
    print("🧪 Starting Nordkapp Aurora Test")

    # Darkness gate
    dark = hour_darkness_gate(SUN_SCHEDULE_FILE)
    print(f"Darkness gate: dark={dark}")
    if not dark:
        print("⏭️ Not dark at Nordkapp — skipping test.")
        return

    # KP gate (multi-source)
    kp_potential = kp_gate_via_script()
    print(f"KP potential gate: {kp_potential}")
    if not kp_potential:
        print("⏭️ KP / aurora potential below threshold — skipping.")
        return

    # Localized Kp proxy for Nordkapp
    kp_val = fetch_current_kp(lat=NORDKAPP_LAT, lon=NORDKAPP_LON)

    # Webcam image acquisition
    url = load_webcam_url()
    if not url:
        print("❌ No Nordkapp webcam URL found")
        return
    ts = int(time.time())
    img_path = IMAGE_TEMP_DIR / f"nordkapp_{ts}.jpg"
    print(f"Downloading Nordkapp webcam image: {url}")
    if not download_image(url, img_path):
        print("❌ Failed to download Nordkapp webcam image")
        return

    try:
        # Color heuristic
        if not check_for_aurora_colors(img_path):
            print("No strong aurora-like color signal — ending test.")
            return
        print("Color heuristic positive — asking AI for verification...")
        ai_answer = ask_ai_if_aurora(img_path, "Nordkapp")
        if ai_answer == "yes":
            print("✅ AI confirms aurora — posting TEST message to Bluesky")
            posted = post_to_bluesky(img_path, "Nordkapp Plateau (Test)", kp_val, test=True)
            print(f"Bluesky post success={posted}")
        elif ai_answer == "no":
            print("AI denies aurora (false positive from heuristic). Test complete.")
        else:
            print("AI inconclusive — skipping post.")
    finally:
        try:
            img_path.unlink()
        except Exception:
            pass

    print("🧪 Nordkapp test finished.")

if __name__ == "__main__":
    main()
