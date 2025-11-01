#!/usr/bin/env python3
"""
Quick darkness check using cached sun schedule.
Returns exit code 0 if dark, 1 if not dark.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def is_dark_now() -> bool:
    """Check if it's currently dark based on cached sun schedule."""
    schedule_file = Path("sun_schedule.json")
    
    if not schedule_file.exists():
        print("⚠️  sun_schedule.json not found; assuming dark (fail-open)", file=sys.stderr)
        return True
    
    try:
        with open(schedule_file, 'r') as f:
            schedule = json.load(f)
        
        sun_times = schedule.get("sun_times", {})
        if not sun_times:
            print("⚠️  Invalid sun_schedule.json; assuming dark (fail-open)", file=sys.stderr)
            return True
        
        now = datetime.now(timezone.utc)
        
        # Parse twilight times
        def parse_time(time_str):
            return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        
        twilight_begin = parse_time(sun_times["astronomical_twilight_begin"])
        twilight_end = parse_time(sun_times["astronomical_twilight_end"])
        
        # Dark if outside twilight period
        is_dark = now < twilight_begin or now > twilight_end
        
        if is_dark:
            print(f"✅ Dark (astronomical night)")
        else:
            print(f"☀️  Not dark enough (astronomical twilight or daylight)")
        
        return is_dark
        
    except Exception as e:
        print(f"⚠️  Error reading sun schedule: {e}; assuming dark (fail-open)", file=sys.stderr)
        return True

if __name__ == "__main__":
    sys.exit(0 if is_dark_now() else 1)
