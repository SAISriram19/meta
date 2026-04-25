"""Core StakeholderEnv — the OpenEnv-compliant environment.

Lifecycle:
    reset(task_id, difficulty) -> Observation
    step(action)               -> StepResult
    state()                    -> EnvironmentState

Each step:
    1. Apply the agent action to the world / memory.
    2. Advance step counter; stakeholders emit inbound messages for next step.
    3. Auto-write new inbound to episodic memory.
    4. Update metrics from stakeholder satisfaction + decisions.
    5. Grade (step reward) and check termination.
    6. Build observation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from env.critic import build_critic
from env.grader import Grader, RewardBreakdown
from env.memory import MemoryStore
from env.models import (
    Action,
    ActionType,
    AllocateAction,
    DecisionPoint,
    EnvironmentState,
    EpisodicMemory,
    ForgetAction,
    GroundTruthTag,
    HiddenState,
    LinkMemoryAction,
    MemoryUpdateAction,
    MessageType,
    Observation,
    ProjectState,
    QueryMemoryAction,
    ReflectAction,
    Scenario,
    SemanticMemory,
    SendMessageAction,
    StakeholderMessage,
    StakeholderSpec,
    StepResult,
    SubmitAction,
    TakeDecisionAction,
    WaitAction,
)
from env.stakeholders import StakeholderPool


SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


class StakeholderEnv:
    """Long-horizon, multi-stakeholder, sycophancy-aware training environment."""

    def __init__(
        self,
        scenario_registry: dict[str, Scenario] | None = None,
        scenario_dir: Path | None = None,
        critic_mode: str = "rules",      # "rules" | "llm" | "none"
    ):
        self.scenario_dir = scenario_dir or SCENARIO_DIR
        self.scenarios: dict[str, Scenario] = scenario_registry or {}
        if not self.scenarios:
            self._load_scenarios_from_disk()
        self.critic_mode = critic_mode
        # Episode state — set by reset().
        self.scenario: Scenario | None = None
        self.state: ProjectState | None = None
        self.hidden: HiddenState | None = None
        self.pool: StakeholderPool | None = None
        self.memory: MemoryStore | None = None
        self.grader: Grader | None = None
        self.pending_inbound: list[StakeholderMessage] = []
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        self.agent_utterances: list[str] = []
        self.last_memory_hits: list[EpisodicMemory | SemanticMemory] = []
        self.drift_applied_this_step: bool = False
        self.semantic_rules_used: set[str] = set()

    # ------------------------------------------------------------------ #
    # Scenario loading                                                   #
    # ------------------------------------------------------------------ #

    def _load_scenarios_from_disk(self):
        if not self.scenario_dir.exists():
            return
        for path in self.scenario_dir.rglob("*.yaml"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                if not isinstance(raw, dict) or "scenario_id" not in raw:
                    continue
                scenario = Scenario.model_validate(raw)
                self.scenarios[scenario.scenario_id] = scenario
            except Exception as e:
                # Generated scenarios can occasionally be malformed; skip noisy.
                print(f"  skipped {path.name}: {e}")

    def register_scenario(self, scenario: Scenario):
        self.scenarios[scenario.scenario_id] = scenario

    def list_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": s.scenario_id,
                "difficulty": s.difficulty_level,
                "steps": s.step_budget,
                "stakeholders": len(s.stakeholders),
            }
            for s in self.scenarios.values()
        ]

    # ------------------------------------------------------------------ #
    # reset                                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        task_id: str | None = None,
        difficulty: int | None = None,
    ) -> Observation:
        scenario = self._pick_scenario(task_id, difficulty)
        self.scenario = scenario

        self.state = ProjectState(
            metrics=dict(scenario.initial_metrics),
            budget_remaining=scenario.initial_budget,
            step=0,
            step_budget=scenario.step_budget,
            stakeholder_satisfaction={s.id: 0.5 for s in scenario.stakeholders},
        )
        self.hidden = HiddenState(
            true_goal=scenario.hidden_true_goal,
            true_metrics=dict(scenario.initial_metrics),
        )
        coord_groups = getattr(scenario, "coordination_groups", None) or []
        self.pool = StakeholderPool(
            scenario.stakeholders,
            adversarial_stakeholder_id=scenario.adversarial_stakeholder_id,
            coordination_groups=coord_groups,
        )
        self.memory = MemoryStore()
        self.memory.tick(0)
        critic = build_critic(self.critic_mode) if self.critic_mode != "none" else None
        self.grader = Grader(critic=critic)
        self.pending_inbound = []
        self.done = False
        self.cumulative_reward = 0.0
        self.agent_utterances = []
        self.agent_stances: list[str] = []
        self.inbound_tags: list = []
        self.queried_memory_ever: bool = False
        self.reflected_ever: bool = False
        self.last_memory_hits = []
        self.drift_applied_this_step = False
        self.semantic_rules_used = set()

        # MemAgent-style rolling working memory. Last MemoryUpdateAction.
        self.working_memory: str = ""
        self.working_memory_facts: list[str] = []
        self.working_memory_history: list[dict] = []  # for reward shaping

        # Seed: stakeholders speak at step 0 before the agent acts.
        self.pending_inbound = self.pool.step(step=0)
        self._write_inbound_to_memory(self.pending_inbound)
        return self._make_observation(feedback="episode started")

    def _pick_scenario(
        self, task_id: str | None, difficulty: int | None
    ) -> Scenario:
        if task_id and task_id in self.scenarios:
            return self.scenarios[task_id]
        if difficulty is not None:
            matching = [s for s in self.scenarios.values() if s.difficulty_level == difficulty]
            if matching:
                return matching[0]
        if self.scenarios:
            return next(iter(self.scenarios.values()))
        raise RuntimeError("No scenarios registered.")

    # ------------------------------------------------------------------ #
    # step                                                               #
    # ------------------------------------------------------------------ #

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode terminated. Call reset().")

        assert self.state and self.hidden and self.pool and self.memory and self.grader

        self.state.step += 1
        step = self.state.step
        self.memory.tick(step)

        addressed_message = self._identify_addressed_message(action)
        semantic_applied: list[str] = []

        # --- Apply action ---
        feedback: str | None = None
        if isinstance(action, SendMessageAction):
            feedback = self._apply_send_message(action, addressed_message)
        elif isinstance(action, TakeDecisionAction):
            feedback = self._apply_decision(action)
        elif isinstance(action, AllocateAction):
            feedback = self._apply_allocate(action)
        elif isinstance(action, QueryMemoryAction):
            feedback = self._apply_query_memory(action, semantic_applied)
        elif isinstance(action, ReflectAction):
            feedback = self._apply_reflect(action)
        elif isinstance(action, LinkMemoryAction):
            feedback = self._apply_link(action)
        elif isinstance(action, ForgetAction):
            feedback = self._apply_forget(action)
        elif isinstance(action, WaitAction):
            feedback = "time passes"
        elif isinstance(action, SubmitAction):
            feedback = "final plan submitted"
            self.done = True
        elif isinstance(action, MemoryUpdateAction):
            feedback = self._apply_memory_update(action)

        # --- Track agent stance history + memory behavior for adversary. ---
        if isinstance(action, SendMessageAction):
            self.agent_stances.append(action.stance.value)
            # tag associated with the message being addressed
            self.inbound_tags.append(
                addressed_message.ground_truth_tag if addressed_message else None
            )
        if isinstance(action, QueryMemoryAction):
            self.queried_memory_ever = True
        if isinstance(action, ReflectAction):
            self.reflected_ever = True

        # --- Stakeholders produce next-step messages (drift applied here) ---
        self.drift_applied_this_step = False
        next_inbound: list[StakeholderMessage] = []
        if not self.done:
            env_summary = self._summarize_env_for_stakeholders()
            next_inbound = self.pool.step(
                step=step,
                recent_agent_utterances=self.agent_utterances[-6:],
                agent_stances=self.agent_stances[-12:],
                inbound_tags=self.inbound_tags[-12:],
                queries_memory=self.queried_memory_ever,
                reflects=self.reflected_ever,
                env_summary=env_summary,
            )
            self._write_inbound_to_memory(next_inbound)
            self.drift_applied_this_step = any(
                len(rt.applied_drift_steps) > 0
                and step in rt.applied_drift_steps
                for rt in self.pool.runtimes.values()
            )

        # --- Metric drift from stakeholder satisfaction ---
        self._update_metrics_from_satisfaction()

        # --- Grade step ---
        memory_hits_for_reward = [
            m for m in self.last_memory_hits if isinstance(m, EpisodicMemory)
        ]
        step_rb: RewardBreakdown = self.grader.step_reward(
            action=action,
            last_inbound=self.pending_inbound,
            addressed_message=addressed_message,
            hidden=self.hidden,
            drift_applied_this_step=self.drift_applied_this_step,
            memory_hits_in_prev_obs=memory_hits_for_reward,
            semantic_applied_ids=semantic_applied,
        )
        self.cumulative_reward += step_rb.total

        terminal_rb: RewardBreakdown | None = None
        if self.done or step >= self.scenario.step_budget:
            self.done = True
            terminal_rb = self.grader.terminal_reward(
                scenario=self.scenario,
                final_state=self.state,
                hidden=self.hidden,
                agent_submitted=isinstance(action, SubmitAction),
            )
            self.cumulative_reward += terminal_rb.total

        # --- Roll inbound forward ---
        self.pending_inbound = next_inbound
        # Clear memory hits unless this step was a QUERY_MEMORY (they're already used).
        if not isinstance(action, QueryMemoryAction):
            self.last_memory_hits = []

        # ACT-R sweep — keep the store from growing unbounded.
        self.memory.sweep_decayed()

        reward_total = step_rb.total + (terminal_rb.total if terminal_rb else 0.0)
        info: dict[str, Any] = {
            "step_reward_breakdown": step_rb.to_dict(),
            "cumulative_reward": self.cumulative_reward,
            "memory_stats": self.memory.stats(),
        }
        if terminal_rb is not None:
            info["terminal_breakdown"] = terminal_rb.to_dict()

        return StepResult(
            observation=self._make_observation(feedback=feedback),
            reward=reward_total,
            done=self.done,
            info=info,
        )

    # ------------------------------------------------------------------ #
    # Action handlers                                                    #
    # ------------------------------------------------------------------ #

    def _apply_send_message(
        self, action: SendMessageAction, addressed: StakeholderMessage | None
    ) -> str:
        # Track agent utterance for future stakeholder context.
        self.agent_utterances.append(action.content)
        # Write the outbound message to episodic memory too.
        self.memory.write_episode(
            step=self.state.step,
            content=f"[agent→{action.stakeholder_id}] ({action.stance.value}) {action.content}",
            importance=0.6,
        )
        rt = self.pool.get_runtime(action.stakeholder_id)
        if rt is None:
            return f"unknown stakeholder {action.stakeholder_id}"
        # Adjust satisfaction: AGREE nudges up, PUSHBACK/REFUSE nudges down,
        # NEGOTIATE neutral-ish, CLARIFY tiny positive. This is the "surface" signal
        # agents who chase satisfaction will overfit to — and lose real reward.
        delta = {
            "agree": 0.08,
            "pushback": -0.04,
            "refuse": -0.08,
            "negotiate": 0.02,
            "clarify": 0.03,
        }.get(action.stance.value, 0.0)
        sid = action.stakeholder_id
        sat = self.state.stakeholder_satisfaction.get(sid, 0.5)
        self.state.stakeholder_satisfaction[sid] = max(0.0, min(1.0, sat + delta))
        return f"sent {action.stance.value} to {sid}"

    def _apply_decision(self, action: TakeDecisionAction) -> str:
        # Find matching decision point by id.
        dp: DecisionPoint | None = next(
            (d for d in self.scenario.decision_points if d.decision_id == action.decision_id),
            None,
        )
        if dp is None:
            return f"unknown decision {action.decision_id}"
        correct = str(action.value) == dp.hidden_correct_option
        # Nudge the hidden true metric — correct decisions move the needle.
        goal_metric = self.scenario.hidden_true_goal.get("metric")
        if goal_metric:
            nudge = 0.25 if correct else -0.15
            self.state.metrics[goal_metric] = self.state.metrics.get(goal_metric, 0.0) + nudge
            self.hidden.true_metrics[goal_metric] = self.hidden.true_metrics.get(goal_metric, 0.0) + nudge
        self.memory.write_episode(
            step=self.state.step,
            content=f"[decision] {action.decision_id}={action.value} (correct={correct})",
            importance=0.9,
        )
        return f"decision {action.decision_id} recorded"

    def _apply_allocate(self, action: AllocateAction) -> str:
        if self.state.budget_remaining < action.amount:
            return "insufficient budget"
        self.state.budget_remaining -= action.amount
        m = self.state.metrics
        m[action.resource] = m.get(action.resource, 0.0) + action.amount
        self.memory.write_episode(
            step=self.state.step,
            content=f"[allocate] {action.amount} → {action.resource}",
            importance=0.5,
        )
        return f"allocated {action.amount} to {action.resource}"

    def _apply_memory_update(self, action: MemoryUpdateAction) -> str:
        """MemAgent-style working-memory write. Overwrites the rolling summary,
        records the snapshot for reward shaping, and writes a low-importance
        episodic trace so retrieval still works on the original events."""
        summary = (action.rolling_summary or "")[:500]
        self.working_memory = summary
        self.working_memory_facts = list(action.key_facts or [])[:10]
        self.working_memory_history.append({
            "step": self.state.step,
            "summary": summary,
            "facts": list(self.working_memory_facts),
        })
        # Trace to episodic store so the agent can later QUERY this summary.
        self.memory.write_episode(
            step=self.state.step,
            content=f"[working_memory] {summary}",
            importance=0.4,
            cues=["working_memory"] + self.working_memory_facts[:5],
        )
        return f"working memory updated ({len(summary)} chars, {len(self.working_memory_facts)} facts)"

    def _apply_query_memory(
        self,
        action: QueryMemoryAction,
        semantic_applied: list[str],
    ) -> str:
        hits = self.memory.query(
            query_text=action.query,
            cues=action.cues or None,
            top_k=action.top_k,
        )
        self.last_memory_hits = hits
        for m in hits:
            if isinstance(m, SemanticMemory):
                if m.memory_id not in self.semantic_rules_used:
                    self.semantic_rules_used.add(m.memory_id)
                    semantic_applied.append(m.memory_id)
        return f"retrieved {len(hits)} memories"

    def _apply_reflect(self, action: ReflectAction) -> str:
        # Pull episodic ids within span range.
        derived_from = [
            m.memory_id
            for m in self.memory.episodic.values()
            if action.span_start <= m.step <= action.span_end
        ]
        if not derived_from:
            return "reflect: no memories in span"
        sem = self.memory.write_semantic(
            step=self.state.step,
            rule=action.rule,
            derived_from=derived_from,
        )
        return f"reflected into {sem.memory_id}"

    def _apply_link(self, action: LinkMemoryAction) -> str:
        ok = self.memory.link(action.memory_a, action.memory_b, action.relation)
        return "linked" if ok else "link failed"

    def _apply_forget(self, action: ForgetAction) -> str:
        ok = self.memory.forget(action.memory_id)
        return "forgotten" if ok else "memory not found"

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _identify_addressed_message(
        self, action: Action
    ) -> StakeholderMessage | None:
        if not isinstance(action, SendMessageAction):
            return None
        for m in reversed(self.pending_inbound):
            if m.stakeholder_id == action.stakeholder_id:
                return m
        return self.pending_inbound[-1] if self.pending_inbound else None

    def _write_inbound_to_memory(self, msgs: list[StakeholderMessage]):
        for m in msgs:
            importance = 0.7 if m.message_type in (
                MessageType.MANIPULATION, MessageType.LIE, MessageType.PRESSURE
            ) else 0.5
            self.memory.write_episode(
                step=m.step,
                content=f"[{m.stakeholder_id}] {m.content}",
                importance=importance,
            )

    def _update_metrics_from_satisfaction(self):
        # Stakeholder satisfaction nudges observable metrics, weighted by influence.
        # This creates the sycophancy trap: chasing satisfaction moves the SURFACE
        # metric, while hidden_true_metrics is what terminal reward checks.
        avg_sat = sum(self.state.stakeholder_satisfaction.values()) / max(
            1, len(self.state.stakeholder_satisfaction)
        )
        self.state.metrics["stakeholder_health"] = avg_sat

    def _summarize_env_for_stakeholders(self) -> str:
        m = self.state.metrics
        return (
            f"step {self.state.step}/{self.state.step_budget}, "
            f"budget {self.state.budget_remaining:.1f}, "
            f"key metrics: {dict((k, round(v, 2)) for k, v in m.items() if k != 'stakeholder_health')}"
        )

    def _make_observation(self, feedback: str | None = None) -> Observation:
        return Observation(
            new_messages=self.pending_inbound,
            state_snapshot=self.state,
            memory_hits=self.last_memory_hits,
            time_remaining=self.state.step_budget - self.state.step,
            last_action_feedback=feedback,
        )

    def get_state(self, debug: bool = False) -> EnvironmentState:
        return EnvironmentState(
            scenario_id=self.scenario.scenario_id if self.scenario else "",
            difficulty_level=self.scenario.difficulty_level if self.scenario else 0,
            step=self.state.step if self.state else 0,
            step_budget=self.state.step_budget if self.state else 0,
            cumulative_reward=self.cumulative_reward,
            project_state=self.state or ProjectState(budget_remaining=0, step=0, step_budget=0),
            episodic_count=len(self.memory.episodic) if self.memory else 0,
            semantic_count=len(self.memory.semantic) if self.memory else 0,
            done=self.done,
            hidden=self.hidden if debug else None,
        )
