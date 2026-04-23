"""Pydantic models for Stakeholder Management Gym.

All data structures that cross the client/server boundary or persist between
steps live here. Keep them minimal and explicit — the LLM agent will see many
of these serialized, so field names matter.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums                                                                       #
# --------------------------------------------------------------------------- #


class Stance(str, Enum):
    AGREE = "agree"
    PUSHBACK = "pushback"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    NEGOTIATE = "negotiate"


class ActionType(str, Enum):
    SEND_MESSAGE = "send_message"
    TAKE_DECISION = "take_decision"
    ALLOCATE = "allocate"
    QUERY_MEMORY = "query_memory"
    REFLECT = "reflect"
    LINK_MEMORY = "link_memory"
    FORGET = "forget"
    WAIT = "wait"
    SUBMIT = "submit"


class MessageType(str, Enum):
    REASONABLE = "reasonable"
    REQUEST = "request"
    COMPLAINT = "complaint"
    MANIPULATION = "manipulation"
    LIE = "lie"
    PRESSURE = "pressure"


class GroundTruthTag(str, Enum):
    GOOD = "good"        # agreeing is correct
    BAD = "bad"          # agreeing is sycophantic
    NEUTRAL = "neutral"  # either is defensible


class RelationType(str, Enum):
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    FOLLOWS = "follows"
    REFERENCES = "references"


# --------------------------------------------------------------------------- #
# Messages                                                                    #
# --------------------------------------------------------------------------- #


class StakeholderMessage(BaseModel):
    """Inbound message from a stakeholder to the agent."""

    step: int
    stakeholder_id: str
    content: str
    message_type: MessageType
    # Hidden fields — not serialized to the agent, used for grading.
    ground_truth_tag: GroundTruthTag | None = Field(default=None, exclude=True)
    manipulation_pattern: str | None = Field(default=None, exclude=True)

    def to_agent_view(self) -> dict[str, Any]:
        """Return only fields the agent is allowed to see."""
        return {
            "step": self.step,
            "from": self.stakeholder_id,
            "content": self.content,
            "type": self.message_type.value,
        }


# --------------------------------------------------------------------------- #
# Actions (agent → env)                                                       #
# --------------------------------------------------------------------------- #


class SendMessageAction(BaseModel):
    type: ActionType = ActionType.SEND_MESSAGE
    stakeholder_id: str
    content: str
    stance: Stance
    reasoning: str | None = None  # optional justification, used by critic


class TakeDecisionAction(BaseModel):
    type: ActionType = ActionType.TAKE_DECISION
    decision_id: str
    value: str | int | float | bool


class AllocateAction(BaseModel):
    type: ActionType = ActionType.ALLOCATE
    resource: str
    amount: float


class QueryMemoryAction(BaseModel):
    type: ActionType = ActionType.QUERY_MEMORY
    query: str
    cues: list[str] = Field(default_factory=list)
    top_k: int = 5


class ReflectAction(BaseModel):
    """Consolidate a span of episodic memories into a semantic rule."""

    type: ActionType = ActionType.REFLECT
    span_start: int
    span_end: int
    rule: str  # agent-authored; grader rewards future applicability


class LinkMemoryAction(BaseModel):
    type: ActionType = ActionType.LINK_MEMORY
    memory_a: str
    memory_b: str
    relation: RelationType


class ForgetAction(BaseModel):
    type: ActionType = ActionType.FORGET
    memory_id: str


class WaitAction(BaseModel):
    type: ActionType = ActionType.WAIT


class SubmitAction(BaseModel):
    type: ActionType = ActionType.SUBMIT
    final_plan: str


Action = (
    SendMessageAction
    | TakeDecisionAction
    | AllocateAction
    | QueryMemoryAction
    | ReflectAction
    | LinkMemoryAction
    | ForgetAction
    | WaitAction
    | SubmitAction
)


# --------------------------------------------------------------------------- #
# Memory                                                                      #
# --------------------------------------------------------------------------- #


class EpisodicMemory(BaseModel):
    """A timestamped event with salience. Hippocampal-trace analogue."""

    memory_id: str
    step: int
    content: str
    cues: list[str] = Field(default_factory=list)      # retrieval keys
    importance: float = 0.5                             # [0, 1]
    embedding: list[float] | None = None                # populated by server
    links: list[str] = Field(default_factory=list)     # ids of linked memories
    decay: float = 1.0                                  # ACT-R retention factor


class SemanticMemory(BaseModel):
    """An abstracted rule derived via REFLECT. Consolidated pattern."""

    memory_id: str
    rule: str
    derived_from: list[str]                             # episodic memory ids
    created_step: int
    applications: int = 0                               # used N times later


MemoryEntry = EpisodicMemory | SemanticMemory


# --------------------------------------------------------------------------- #
# World state                                                                 #
# --------------------------------------------------------------------------- #


class ProjectState(BaseModel):
    """Public, observable project metrics. Updated by TAKE_DECISION/ALLOCATE."""

    metrics: dict[str, float] = Field(default_factory=dict)
    budget_remaining: float
    step: int
    step_budget: int
    stakeholder_satisfaction: dict[str, float] = Field(default_factory=dict)


class HiddenState(BaseModel):
    """Never exposed to the agent. Drives scoring and stakeholder behavior."""

    true_goal: dict[str, Any]
    true_metrics: dict[str, float] = Field(default_factory=dict)
    bad_agreements: list[int] = Field(default_factory=list)     # step ids
    principled_pushbacks: list[int] = Field(default_factory=list)
    caught_manipulations: list[int] = Field(default_factory=list)
    memory_use_hits: list[int] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Observations (env → agent)                                                  #
# --------------------------------------------------------------------------- #


class Observation(BaseModel):
    new_messages: list[StakeholderMessage] = Field(default_factory=list)
    state_snapshot: ProjectState
    memory_hits: list[EpisodicMemory | SemanticMemory] = Field(default_factory=list)
    time_remaining: int
    last_action_feedback: str | None = None     # lightweight env response

    def to_agent_view(self) -> dict[str, Any]:
        return {
            "messages": [m.to_agent_view() for m in self.new_messages],
            "state": self.state_snapshot.model_dump(),
            "memory_hits": [_memory_to_agent_view(m) for m in self.memory_hits],
            "time_remaining": self.time_remaining,
            "feedback": self.last_action_feedback,
        }


def _memory_to_agent_view(m) -> dict[str, Any]:
    """Serialize a memory entry for the agent — strip heavy fields (embedding)."""
    if isinstance(m, EpisodicMemory):
        return {
            "id": m.memory_id,
            "kind": "episodic",
            "step": m.step,
            "content": m.content,
            "cues": m.cues[:6],
            "importance": round(m.importance, 2),
            "links": m.links[:4],
        }
    if isinstance(m, SemanticMemory):
        return {
            "id": m.memory_id,
            "kind": "semantic",
            "rule": m.rule,
            "derived_from": m.derived_from[:4],
            "applications": m.applications,
        }
    return {}


class StepResult(BaseModel):
    """Standard (obs, reward, done, info) shape for OpenEnv."""

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Scenarios                                                                   #
# --------------------------------------------------------------------------- #


class DriftEvent(BaseModel):
    step: int
    new_stance: str
    new_hidden_preference: dict[str, Any] | None = None


class ScriptedMessage(BaseModel):
    """A pre-authored message a stakeholder will send at a given step."""

    step: int
    content: str
    message_type: MessageType
    ground_truth_tag: GroundTruthTag
    manipulation_pattern: str | None = None


class StakeholderSpec(BaseModel):
    id: str
    name: str
    public_stance: str
    hidden_preferences: dict[str, Any] = Field(default_factory=dict)
    honesty: float = 1.0                          # [0, 1]
    influence: float = 0.5                        # [0, 1]
    drift_schedule: list[DriftEvent] = Field(default_factory=list)
    scripted_messages: list[ScriptedMessage] = Field(default_factory=list)
    persona_prompt: str | None = None             # for LLM-backed personas


class DecisionPoint(BaseModel):
    decision_id: str
    step_range: tuple[int, int]
    options: list[str]
    hidden_correct_option: str                    # never exposed
    reasoning_trace: str | None = None            # hidden; why correct option is correct


class Scenario(BaseModel):
    scenario_id: str
    difficulty_level: int
    step_budget: int
    hidden_true_goal: dict[str, Any]
    initial_metrics: dict[str, float] = Field(default_factory=dict)
    initial_budget: float
    stakeholders: list[StakeholderSpec]
    decision_points: list[DecisionPoint] = Field(default_factory=list)
    adversarial_stakeholder_id: str | None = None     # None at L0-L2
    coordination_groups: list[list[str]] = Field(default_factory=list)
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Full environment state (for debugging / state() endpoint)                   #
# --------------------------------------------------------------------------- #


class EnvironmentState(BaseModel):
    scenario_id: str
    difficulty_level: int
    step: int
    step_budget: int
    cumulative_reward: float
    project_state: ProjectState
    episodic_count: int
    semantic_count: int
    done: bool
    # Hidden shadow — only included when debug=True.
    hidden: HiddenState | None = None
