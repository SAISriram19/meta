"""FastAPI entrypoint for the Stakeholder Management Gym environment.

Single-episode-per-process model (matches OpenEnv reference pattern):
    GET  /health                 → 200 OK
    GET  /tasks                  → list scenarios
    POST /reset                  → begin new episode, return first observation
    POST /step                   → take one step, return (obs, reward, done, info)
    GET  /state                  → current environment state (optionally debug)

Hidden ground-truth is only exposed when `?debug=true` is passed on /state,
and is NEVER included in /reset or /step responses.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from env.environment import StakeholderEnv
from server.schemas import (
    ActionRequest,
    ResetRequest,
    StateResponse,
    StepResponse,
    TaskListResponse,
)


app = FastAPI(
    title="Stakeholder Management Gym",
    description=(
        "Long-horizon, multi-stakeholder RL environment for training LLMs "
        "against sycophancy. OpenEnv Round 2 submission."
    ),
    version="0.1.0",
)

# Single shared env — one active episode per process.
env = StakeholderEnv()


@app.get("/health")
def health():
    return {"status": "ok", "scenarios_loaded": len(env.scenarios)}


@app.get("/tasks", response_model=TaskListResponse)
def tasks():
    return TaskListResponse(tasks=env.list_tasks())


@app.post("/reset")
def reset(req: ResetRequest | None = None):
    try:
        obs = env.reset(
            task_id=req.task_id if req else None,
            difficulty=req.difficulty if req else None,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(
        content={
            "observation": obs.to_agent_view(),
            "scenario_id": env.scenario.scenario_id if env.scenario else None,
            "step_budget": env.state.step_budget if env.state else 0,
            "difficulty": env.scenario.difficulty_level if env.scenario else 0,
        }
    )


@app.post("/step", response_model=StepResponse)
def step(req: ActionRequest):
    try:
        action = req.to_action()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StepResponse(
        observation=result.observation.to_agent_view(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state", response_model=StateResponse)
def state(debug: bool = Query(False)):
    s = env.get_state(debug=debug)
    return StateResponse(state=s.model_dump())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
