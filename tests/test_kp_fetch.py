import json
from types import SimpleNamespace
from src.main import fetch_current_kp
import requests

class DummyResp:
    def __init__(self, data):
        self._data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self._data


def test_fetch_current_kp_exact_match(monkeypatch):
    data = {"coordinates": [[59,10,3.7],[60,11,1.2]]}
    def fake_get(url, timeout=12):
        return DummyResp(data)
    monkeypatch.setattr(requests, 'get', fake_get)
    val = fetch_current_kp(59.1,10.2)
    assert val == 3.7


def test_fetch_current_kp_nearest(monkeypatch):
    data = {"coordinates": [[58.5,9.5,2.5],[58.9,9.9,4.2]]}
    def fake_get(url, timeout=12):
        return DummyResp(data)
    monkeypatch.setattr(requests, 'get', fake_get)
    val = fetch_current_kp(59.0,10.0)
    assert val == 4.2


def test_fetch_current_kp_none(monkeypatch):
    data = {"coordinates": []}
    def fake_get(url, timeout=12):
        return DummyResp(data)
    monkeypatch.setattr(requests, 'get', fake_get)
    val = fetch_current_kp(59.0,10.0)
    assert val is None
