from pathlib import Path

import pytest

import src.main as main


@pytest.mark.parametrize(("response", "expected"), [("yes", "yes"), ("No.", "no")])
def test_ask_ai_if_aurora_uses_copilot_cli(monkeypatch, tmp_path, response, expected):
    image_path = tmp_path / "aurora.jpg"
    image_path.write_bytes(b"image")
    monkeypatch.setenv("GH_TOKEN", "dummy-token")
    monkeypatch.setattr(main.shutil, "which", lambda command: "/usr/bin/copilot")

    def fake_run(command, **kwargs):
        assert command[:3] == ["copilot", "--model", "auto"]
        assert command[command.index("--attachment") + 1] == str(image_path)
        assert kwargs["env"]["GH_TOKEN"] == "dummy-token"
        assert kwargs["timeout"] == 60
        return main.subprocess.CompletedProcess(command, 0, response, "")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    assert main.ask_ai_if_aurora(image_path, "TestLoc") == expected


def test_ask_ai_if_aurora_without_token_skips_cli(monkeypatch, tmp_path):
    image_path = tmp_path / "aurora.jpg"
    image_path.write_bytes(b"image")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_PAT", raising=False)
    monkeypatch.setattr(main.shutil, "which", lambda command: "/usr/bin/copilot")
    monkeypatch.setattr(
        main.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("Copilot CLI should not run without a token"),
    )

    assert main.ask_ai_if_aurora(image_path, "TestLoc") is None
