"""HTTP request/response envelopes for the FastAPI server."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from env.models import (
    Action,
    ActionType,
    AllocateAction,
    ForgetAction,
    LinkMemoryAction,
    QueryMemoryAction,
    ReflectAction,
    RelationType,
    SendMessageAction,
    Stance,
    SubmitAction,
    TakeDecisionAction,
    WaitAction,
)


class ResetRequest(BaseModel):
    task_id: str | None = None
    difficulty: int | None = None


class ActionRequest(BaseModel):
    """Single wire-format payload that maps onto any of the Action variants."""

    type: ActionType
    # SendMessage
    stakeholder_id: str | None = None
    content: str | None = None
    stance: Stance | None = None
    reasoning: str | None = None
    # TakeDecision
    decision_id: str | None = None
    value: str | int | float | bool | None = None
    # Allocate
    resource: str | None = None
    amount: float | None = None
    # QueryMemory
    query: str | None = None
    cues: list[str] | None = None
    top_k: int | None = None
    # Reflect
    span_start: int | None = None
    span_end: int | None = None
    rule: str | None = None
    # LinkMemory
    memory_a: str | None = None
    memory_b: str | None = None
    relation: RelationType | None = None
    # Forget
    memory_id: str | None = None
    # Submit
    final_plan: str | None = None

    def to_action(self) -> Action:
        t = self.type
        if t == ActionType.SEND_MESSAGE:
            return SendMessageAction(
                stakeholder_id=self.stakeholder_id or "",
                content=self.content or "",
                stance=self.stance or Stance.CLARIFY,
                reasoning=self.reasoning,
            )
        if t == ActionType.TAKE_DECISION:
            return TakeDecisionAction(
                decision_id=self.decision_id or "",
                value=self.value if self.value is not None else "",
            )
        if t == ActionType.ALLOCATE:
            return AllocateAction(
                resource=self.resource or "",
                amount=float(self.amount or 0.0),
            )
        if t == ActionType.QUERY_MEMORY:
            return QueryMemoryAction(
                query=self.query or "",
                cues=self.cues or [],
                top_k=self.top_k or 5,
            )
        if t == ActionType.REFLECT:
            return ReflectAction(
                span_start=self.span_start or 0,
                span_end=self.span_end or 0,
                rule=self.rule or "",
            )
        if t == ActionType.LINK_MEMORY:
            return LinkMemoryAction(
                memory_a=self.memory_a or "",
                memory_b=self.memory_b or "",
                relation=self.relation or RelationType.REFERENCES,
            )
        if t == ActionType.FORGET:
            return ForgetAction(memory_id=self.memory_id or "")
        if t == ActionType.WAIT:
            return WaitAction()
        if t == ActionType.SUBMIT:
            return SubmitAction(final_plan=self.final_plan or "")
        raise ValueError(f"unknown action type: {t}")


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    state: dict[str, Any]


class TaskListResponse(BaseModel):
    tasks: list[dict[str, Any]]
