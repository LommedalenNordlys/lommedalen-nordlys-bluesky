import base64
from pathlib import Path
from PIL import Image
import io
import requests
import os
from src.main import get_image_data_url, ask_ai_if_aurora, azure_rate_limited

class DummyResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"choices": [{"message": {"content": "yes"}}]}
        self.text = "ok"
    def json(self):
        return self._payload


def make_image(path: Path, size=(1200, 900), color=(10, 150, 40)):
    img = Image.new('RGB', size, color)
    img.save(path, format='JPEG')


def test_get_image_data_url_resizes_and_prefix(tmp_path):
    p = tmp_path / "big.jpg"
    make_image(p)
    data_url = get_image_data_url(p, max_size=(800, 600))
    assert data_url.startswith("data:image/jpeg;base64,")
    # decode and check that image size is within bounds
    encoded = data_url.split(",",1)[1]
    raw = base64.b64decode(encoded)
    im = Image.open(io.BytesIO(raw))
    assert im.size[0] <= 800 and im.size[1] <= 600


def test_ask_ai_if_aurora_yes(monkeypatch, tmp_path):
    # Prepare image
    p = tmp_path / "aurora.jpg"
    make_image(p, size=(400,300), color=(20,180,50))

    # Force TOKEN so function proceeds
    os.environ['KEY_GITHUB_TOKEN'] = 'dummy-token'
    # Reset rate limit flag
    global azure_rate_limited
    azure_rate_limited = False

    def fake_post(url, json=None, headers=None, timeout=30):
        return DummyResp(200, {"choices": [{"message": {"content": "yes"}}]})
    monkeypatch.setattr(requests, 'post', fake_post)
    answer = ask_ai_if_aurora(p, "TestLoc")
    assert answer == 'yes'


def test_ask_ai_if_aurora_no(monkeypatch, tmp_path):
    p = tmp_path / "noaurora.jpg"
    make_image(p, size=(400,300), color=(10,10,10))
    os.environ['KEY_GITHUB_TOKEN'] = 'dummy-token'
    global azure_rate_limited
    azure_rate_limited = False

    def fake_post(url, json=None, headers=None, timeout=30):
        return DummyResp(200, {"choices": [{"message": {"content": "no"}}]})
    monkeypatch.setattr(requests, 'post', fake_post)
    answer = ask_ai_if_aurora(p, "TestLoc")
    assert answer == 'no'
