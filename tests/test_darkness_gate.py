import json, tempfile
from pathlib import Path
from datetime import datetime, UTC
from src.nordkapp_test import hour_darkness_gate


def make_schedule(start_hour: int, end_hour: int, spans_midnight: bool = True) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    data = {
        "dark_hours": {
            "start_hour": start_hour,
            "end_hour": end_hour,
            "spans_midnight": spans_midnight
        }
    }
    tmp.write(json.dumps(data).encode())
    tmp.flush()
    return Path(tmp.name)


def test_darkness_gate_midnight_span_dark_hour():
    # Force current hour inside dark range: pick hour that is guaranteed inside
    now_hour = int(datetime.now(UTC).strftime("%H"))
    # Choose start two hours before now and end two hours after (spans midnight logic triggers by condition start>=end)
    start = (now_hour - 2) % 24
    end = (now_hour + 2) % 24
    # Ensure spans_midnight by making start > end
    if start <= end:
        start = (end + 1) % 24
    sched = make_schedule(start, end, True)
    assert hour_darkness_gate(sched) is True


def test_darkness_gate_midnight_span_not_dark():
    # Pick schedule where current hour is explicitly outside window
    now_hour = int(datetime.now(UTC).strftime("%H"))
    start = (now_hour + 2) % 24
    end = (now_hour - 2) % 24
    if start <= end:
        start = (end + 3) % 24
    sched = make_schedule(start, end, True)
    # Current hour should not satisfy (>=start or <=end)
    assert hour_darkness_gate(sched) is False


def test_darkness_gate_linear_period():
    # Non-midnight span: start < end
    now_hour = int(datetime.now(UTC).strftime("%H"))
    start = (now_hour - 1) % 24
    end = (now_hour + 1) % 24
    sched = make_schedule(start, end, False)
    assert hour_darkness_gate(sched) is True


def test_darkness_gate_linear_period_outside():
    now_hour = int(datetime.now(UTC).strftime("%H"))
    start = (now_hour + 1) % 24
    end = (now_hour + 3) % 24
    sched = make_schedule(start, end, False)
    assert hour_darkness_gate(sched) is False
