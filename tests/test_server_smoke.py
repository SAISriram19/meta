"""Smoke test the FastAPI server via TestClient — no network, no uvicorn."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from server.main import app  # noqa: E402


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_tasks_lists_scenarios():
    client = TestClient(app)
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    assert len(tasks) >= 1
    assert any(t["task_id"] == "L0_launch" for t in tasks)


def test_reset_step_loop():
    client = TestClient(app)
    r = client.post("/reset", json={"task_id": "L0_launch"})
    assert r.status_code == 200, r.text
    obs = r.json()["observation"]
    assert "messages" in obs
    # send a pushback to whoever just spoke
    first = obs["messages"][-1] if obs["messages"] else None
    if first:
        action = {
            "type": "send_message",
            "stakeholder_id": first["from"],
            "content": "Let's slow down and do this right.",
            "stance": "pushback",
            "reasoning": "fabricated urgency",
        }
        r = client.post("/step", json=action)
        assert r.status_code == 200, r.text
        assert "reward" in r.json()

    # verify hidden state NOT leaked on regular /state
    r = client.get("/state")
    assert r.status_code == 200
    assert r.json()["state"].get("hidden") is None

    # and IS present on debug
    r = client.get("/state?debug=true")
    assert r.json()["state"].get("hidden") is not None


if __name__ == "__main__":
    test_health()
    test_tasks_lists_scenarios()
    test_reset_step_loop()
    print("server smoke tests passed.")
