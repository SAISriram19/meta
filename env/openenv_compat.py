"""OpenEnv-compliant adapter for the Stakeholder Management Gym.

Wraps our internal `StakeholderEnv` (which has its own Pydantic models) with
the official `openenv.core` base classes, so the env can be used via:

    from env.openenv_compat import StakeholderManagementGym, StakeholderAction
    env = StakeholderManagementGym()
    obs = env.reset()
    action = StakeholderAction(action_json='{"type":"wait"}')
    obs = env.step(action)

Or via the openenv FastAPI server factory:

    from openenv.core import create_fastapi_app
    from env.openenv_compat import StakeholderManagementGym, StakeholderAction, StakeholderObservation
    app = create_fastapi_app(StakeholderManagementGym, StakeholderAction, StakeholderObservation)

Existing `server/main.py` continues to work — this is an alternative entry that
uses the official openenv package.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from openenv.core import Action, Environment, Observation, State

from env.environment import StakeholderEnv as _InternalEnv
from env.models import (
    Action as _IntAction,
    AllocateAction,
    ForgetAction,
    LinkMemoryAction,
    QueryMemoryAction,
    ReflectAction,
    SendMessageAction,
    Stance,
    SubmitAction,
    TakeDecisionAction,
    WaitAction,
)


# --------------------------------------------------------------------------- #
# OpenEnv-compliant Action / Observation / State                              #
# --------------------------------------------------------------------------- #


class StakeholderAction(Action):
    """OpenEnv Action carrying a JSON-serialized internal Action.

    Why JSON: our internal action union has 9 distinct types with type-tagged
    discriminator. Carrying as JSON keeps the openenv-side schema flat (one
    field) and lets the env deserialize via our existing parser.
    """
    action_json: str = ""

    def to_internal(self) -> _IntAction:
        if not self.action_json or not self.action_json.strip():
            return WaitAction()
        try:
            data = json.loads(self.action_json) if isinstance(self.action_json, str) else self.action_json
        except json.JSONDecodeError:
            return WaitAction()
        return _coerce_action(data)


class StakeholderObservation(Observation):
    """OpenEnv Observation. `payload` carries the full internal observation as
    a dict; convenience top-level fields surface the most-used signals."""
    payload: dict[str, Any] = {}
    new_messages: list[dict[str, Any]] = []
    memory_hits: list[dict[str, Any]] = []
    time_remaining: int = 0


class StakeholderState(State):
    """OpenEnv State summary."""
    scenario_id: str = ""
    step: int = 0
    step_budget: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    payload: dict[str, Any] = {}


def _coerce_action(data: dict[str, Any]) -> _IntAction:
    """Map a {'type': str, ...} dict to the right internal Action subclass."""
    t = (data.get("type") or "").lower()
    if t == "send_message":
        stance_raw = data.get("stance", "agree")
        stance = Stance(stance_raw) if isinstance(stance_raw, str) else stance_raw
        return SendMessageAction(
            stakeholder_id=data.get("stakeholder_id", ""),
            content=data.get("content", ""),
            stance=stance,
            reasoning=data.get("reasoning"),
        )
    if t == "take_decision":
        return TakeDecisionAction(
            decision_id=data.get("decision_id", ""),
            value=data.get("value", ""),
        )
    if t == "allocate":
        return AllocateAction(
            resource=data.get("resource", ""),
            amount=float(data.get("amount", 0.0)),
        )
    if t == "query_memory":
        return QueryMemoryAction(
            query=data.get("query", ""),
            cues=data.get("cues", []),
            top_k=int(data.get("top_k", 5)),
        )
    if t == "reflect":
        return ReflectAction(
            span_start=int(data.get("span_start", 0)),
            span_end=int(data.get("span_end", 0)),
            rule=data.get("rule", ""),
        )
    if t == "link_memory":
        return LinkMemoryAction(
            memory_a=data.get("memory_a", ""),
            memory_b=data.get("memory_b", ""),
            relation=data.get("relation", "related"),
        )
    if t == "forget":
        return ForgetAction(memory_id=data.get("memory_id", ""))
    if t == "submit":
        plan = data.get("final_plan", "")
        if isinstance(plan, dict):
            plan = json.dumps(plan)
        return SubmitAction(final_plan=plan)
    return WaitAction()


# --------------------------------------------------------------------------- #
# OpenEnv-compliant Environment                                               #
# --------------------------------------------------------------------------- #


class StakeholderManagementGym(Environment[StakeholderAction, StakeholderObservation, StakeholderState]):
    """OpenEnv-compliant facade over StakeholderEnv.

    Honors the openenv contract (reset/step/state) while delegating the actual
    simulation to our existing internal env. Lets us use this env via the
    official openenv server factory and HF Hub `AutoEnv.from_env(...)` flow.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False  # internal env has per-instance state

    def __init__(self, default_task: str = "L0_launch", **kwargs):
        super().__init__()
        self._inner = _InternalEnv(**kwargs)
        self._default_task = default_task
        self._last_obs = None

    # --- helpers ---

    def _wrap_obs(self, obs, reward: Optional[float] = None) -> StakeholderObservation:
        payload = obs.model_dump()
        return StakeholderObservation(
            done=self._inner.done,
            reward=reward,
            payload=payload,
            new_messages=[m.model_dump() for m in obs.new_messages],
            memory_hits=[m.model_dump() if hasattr(m, "model_dump") else m for m in (obs.memory_hits or [])],
            time_remaining=int(obs.time_remaining or 0),
        )

    # --- openenv contract ---

    def reset(self, task_id: Optional[str] = None, **kwargs) -> StakeholderObservation:
        task = task_id or self._default_task
        obs = self._inner.reset(task_id=task, **kwargs)
        self._last_obs = obs
        return self._wrap_obs(obs)

    def step(self, action: StakeholderAction) -> StakeholderObservation:
        internal = action.to_internal()
        result = self._inner.step(internal)
        self._last_obs = result.observation
        wrapped = self._wrap_obs(result.observation, reward=float(result.reward))
        # Pass-through env info (terminal_breakdown, etc) via metadata.
        wrapped.metadata = {**(wrapped.metadata or {}), **(result.info or {})}
        return wrapped

    def state(self) -> StakeholderState:
        s = self._inner.get_state(debug=False)
        return StakeholderState(
            scenario_id=getattr(s, "scenario_id", ""),
            step=getattr(s, "step", 0),
            step_budget=getattr(s, "step_budget", 0),
            cumulative_reward=getattr(s, "cumulative_reward", 0.0),
            done=getattr(s, "done", False),
            payload=s.model_dump(),
        )

    def close(self) -> None:
        self._inner = None
        self._last_obs = None


# --------------------------------------------------------------------------- #
# Convenience: openenv FastAPI app factory                                    #
# --------------------------------------------------------------------------- #


def create_openenv_app(max_concurrent_envs: Optional[int] = None):
    """Build a FastAPI app using the official openenv server factory."""
    from openenv.core import create_fastapi_app
    return create_fastapi_app(
        env=StakeholderManagementGym,
        action_cls=StakeholderAction,
        observation_cls=StakeholderObservation,
        max_concurrent_envs=max_concurrent_envs,
    )
