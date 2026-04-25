"""OpenEnv-SDK-blessed FastAPI entrypoint.

Uses `openenv.core.create_fastapi_app` so the env exposes the canonical SDK
endpoints (reset, step, state, health, metadata, schema, mcp, ws) without us
re-implementing the protocol. Adds our scenario-listing endpoint on top.

Run:
    uvicorn server.openenv_main:app --host 0.0.0.0 --port 7860

Existing `server/main.py` remains as a legacy entrypoint for back-compat.
"""

from __future__ import annotations

from env.openenv_compat import create_openenv_app
from env.environment import StakeholderEnv

app = create_openenv_app()

# Custom endpoint: scenario list. OpenEnv's create_fastapi_app doesn't include
# this; we add it because the existing client + tests rely on it.
_inspector_env = StakeholderEnv()


@app.get("/tasks")
def tasks():
    """List available scenarios (custom endpoint, not part of OpenEnv core)."""
    return {"tasks": _inspector_env.list_tasks()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
