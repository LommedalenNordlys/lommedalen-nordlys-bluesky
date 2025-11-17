import re
from src.main import build_post_text

def test_build_post_text_normal():
    text = build_post_text("Lommedalen", 2.34, False)
    assert "Lommedalen" in text
    assert "Kp index: 2.3" in text  # rounded to one decimal
    assert "Test" not in text.splitlines()[0]


def test_build_post_text_test_flag():
    text = build_post_text("Nordkapp", 5.0, True)
    lines = text.splitlines()
    assert lines[0].startswith("🧪 Test")
    assert "Nordkapp" in text
    assert "(This is a test post" in text


def test_build_post_text_none_kp():
    text = build_post_text("Somewhere", None, False)
    assert "Kp index: n/a" in text
