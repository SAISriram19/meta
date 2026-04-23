"""HTTP integration test — exercises the OpenEnv HTTP contract end-to-end.

Uses FastAPI's TestClient (no real port binding), but every call goes through
the actual request/response pipeline, serialization, and routing. If this
passes, a running container will serve agents correctly too.

Covers:
    * GET /health
    * GET /tasks
    * POST /reset
    * POST /step   for every action type
    * GET /state   with and without debug
    * Hidden-state leak guards
    * Episode termination contract
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from server.main import app  # noqa: E402


def test_full_http_loop():
    client = TestClient(app)

    # health
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["scenarios_loaded"] >= 2

    # tasks
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    ids = {t["task_id"] for t in tasks}
    assert "L0_launch" in ids
    assert "L2_strategic_shift" in ids

    # reset
    r = client.post("/reset", json={"task_id": "L2_strategic_shift"})
    assert r.status_code == 200
    data = r.json()
    assert data["scenario_id"] == "L2_strategic_shift"
    obs = data["observation"]
    assert "messages" in obs
    assert "state" in obs
    assert "memory_hits" in obs
    assert "time_remaining" in obs

    # No ground-truth leak on the observation.
    for m in obs["messages"]:
        assert "ground_truth_tag" not in m
        assert "manipulation_pattern" not in m

    first_from = obs["messages"][-1]["from"] if obs["messages"] else "chen"

    # step — send_message
    r = client.post("/step", json={
        "type": "send_message",
        "stakeholder_id": first_from,
        "content": "Let me verify that with past statements first.",
        "stance": "pushback",
        "reasoning": "moving_goalposts pattern",
    })
    assert r.status_code == 200
    body = r.json()
    assert "reward" in body
    assert "done" in body
    assert "observation" in body
    assert "info" in body
    assert "step_reward_breakdown" in body["info"]

    # step — query_memory
    r = client.post("/step", json={
        "type": "query_memory",
        "query": "earlier stakeholder statements",
        "cues": ["earlier", "agreed", "committed"],
        "top_k": 3,
    })
    assert r.status_code == 200
    obs = r.json()["observation"]
    # After a query_memory, memory_hits may be populated.
    assert "memory_hits" in obs

    # step — reflect
    r = client.post("/step", json={
        "type": "reflect",
        "span_start": 0,
        "span_end": 5,
        "rule": "Chen escalates when under deadline pressure",
    })
    assert r.status_code == 200

    # step — take_decision (try first decision_id)
    state = client.get("/state").json()["state"]
    # The /state endpoint should NOT include hidden by default.
    assert state.get("hidden") is None

    # decisions list isn't exposed via /state; just pick a known one from scenario.
    r = client.post("/step", json={
        "type": "take_decision",
        "decision_id": "compliance_window",
        "value": "full_three_weeks",
    })
    assert r.status_code == 200

    # step — allocate
    r = client.post("/step", json={
        "type": "allocate",
        "resource": "qa_hours",
        "amount": 5.0,
    })
    assert r.status_code == 200

    # step — wait
    r = client.post("/step", json={"type": "wait"})
    assert r.status_code == 200

    # debug state exposes hidden
    r = client.get("/state?debug=true")
    assert r.status_code == 200
    hidden = r.json()["state"].get("hidden")
    assert hidden is not None
    assert "true_goal" in hidden

    # Run to completion then verify step() returns a 409.
    while True:
        r = client.post("/step", json={"type": "wait"})
        if r.status_code != 200:
            break
        if r.json().get("done"):
            break

    r = client.post("/step", json={"type": "wait"})
    assert r.status_code == 409, "stepping a terminated episode should 409"


def test_reset_between_episodes():
    client = TestClient(app)
    # run first episode to done via fast-waits
    client.post("/reset", json={"task_id": "L0_launch"})
    while True:
        r = client.post("/step", json={"type": "wait"})
        if r.status_code != 200:
            break
        if r.json().get("done"):
            break
    # second reset should work
    r = client.post("/reset", json={"task_id": "L0_launch"})
    assert r.status_code == 200


def test_unknown_task_fails_cleanly():
    client = TestClient(app)
    r = client.post("/reset", json={"task_id": "does_not_exist"})
    # Env falls back to first-available rather than error; verify graceful.
    assert r.status_code == 200


def test_malformed_step_request_rejected():
    client = TestClient(app)
    client.post("/reset", json={"task_id": "L0_launch"})
    r = client.post("/step", json={"type": "not_a_real_action"})
    assert r.status_code == 422  # pydantic validation


if __name__ == "__main__":
    test_full_http_loop()
    print("full http loop passed")
    test_reset_between_episodes()
    print("reset between episodes passed")
    test_unknown_task_fails_cleanly()
    print("unknown task graceful fallback passed")
    test_malformed_step_request_rejected()
    print("malformed step rejection passed")
    print("\nHTTP integration: all green")
