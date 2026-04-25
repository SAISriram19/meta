"""MemoryAgentBench-style 4-competency eval for the Meta env.

Maps to the 4 core memory competencies defined in:
    Hu, Wang & McAuley, "Evaluating Memory in LLM Agents via Incremental
    Multi-Turn Interactions" (ICLR 2026, arXiv 2507.05257)

    AR  — Accurate Retrieval (multi-hop needle-in-haystack)
    TTL — Test-Time Learning (apply newly-introduced rules)
    LRU — Long-Range Understanding (coherent global view)
    CR  — Conflict Resolution (overwrite stale facts on contradictory updates)

Each competency emits a [0, 1] score per rollout, with raw counts for
auditability. Aggregates are mean over seeds × scenarios.

Usage:
    from eval.competencies import score_rollout
    score = score_rollout(env, action_trace)
    # {"AR": 0.83, "TTL": 0.50, "LRU": 0.71, "CR": 0.66}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from env.models import (
    QueryMemoryAction,
    ReflectAction,
    SendMessageAction,
    Stance,
    TakeDecisionAction,
)


# --------------------------------------------------------------------------- #
# Per-step trace record produced during rollout                               #
# --------------------------------------------------------------------------- #


@dataclass
class StepTrace:
    step: int
    inbound_msgs: list[dict] = field(default_factory=list)   # ground-truth-tagged inbound
    action_type: str = ""
    action_data: dict = field(default_factory=dict)
    memory_hits: list[dict] = field(default_factory=list)    # if action was QueryMemory
    reasoning: str = ""
    reward: float = 0.0


# --------------------------------------------------------------------------- #
# Competency 1 — Accurate Retrieval (AR)                                      #
# --------------------------------------------------------------------------- #


def score_AR(trace: list[StepTrace], scenario_id: str) -> dict:
    """AR = how often does QueryMemory return content that ACTUALLY appears in
    a much-earlier inbound message AND gets cited in the agent's next action?

    The 'depth' qualifier matters: retrieving step-50 from step-60 is trivial;
    retrieving step-5 from step-1500 is the test.
    """
    successful_deep_retrievals = 0
    total_queries = 0
    inbound_history = []  # accumulate seen inbound text

    for i, t in enumerate(trace):
        for m in t.inbound_msgs:
            inbound_history.append({"step": t.step, "text": (m.get("content") or "").lower(),
                                    "stakeholder": m.get("stakeholder_id", "")})

        if t.action_type == "query_memory":
            total_queries += 1
            # Check if any retrieved hit corresponds to an inbound at least
            # 50 steps earlier — that's the "deep" retrieval bar.
            depth_threshold = max(50, t.step // 4)
            hit_texts = [(h.get("content") or "").lower()[:200] for h in t.memory_hits]
            for hit in hit_texts:
                if not hit:
                    continue
                # Find earliest inbound that overlaps significantly
                earliest = min(
                    (h["step"] for h in inbound_history
                     if any(w in h["text"] for w in hit.split()[:5] if len(w) > 4)),
                    default=None,
                )
                if earliest is not None and (t.step - earliest) >= depth_threshold:
                    # Check next-step action references the retrieval
                    if i + 1 < len(trace):
                        next_a = trace[i + 1]
                        if next_a.action_type == "send_message":
                            content = (next_a.action_data.get("content") or "").lower()
                            reasoning = (next_a.reasoning or "").lower()
                            # Cite-check: agent's next response shares vocabulary
                            # with the retrieved hit
                            shared = sum(1 for w in hit.split()[:8]
                                         if len(w) > 4 and (w in content or w in reasoning))
                            if shared >= 2:
                                successful_deep_retrievals += 1
                                break  # count each query at most once

    score = successful_deep_retrievals / max(1, total_queries)
    return {
        "score": round(score, 3),
        "successful_deep_retrievals": successful_deep_retrievals,
        "total_queries": total_queries,
        "max_depth_seen": max((t.step for t in trace if t.action_type == "query_memory"), default=0),
    }


# --------------------------------------------------------------------------- #
# Competency 2 — Test-Time Learning (TTL)                                     #
# --------------------------------------------------------------------------- #


def score_TTL(trace: list[StepTrace], scenario_id: str) -> dict:
    """TTL = does the agent emit a REFLECT-style rule, then APPLY it later?

    We map TTL to:
      - REFLECT actions where the `rule` field is non-trivial
      - Subsequent actions whose reasoning matches the rule's keywords
    """
    rules_emitted = []
    rule_applications = 0

    for t in trace:
        if t.action_type == "reflect":
            rule = (t.action_data.get("rule") or "").strip().lower()
            if len(rule.split()) >= 4:
                rules_emitted.append({"step": t.step, "rule": rule})

        elif t.action_type == "send_message":
            reasoning = (t.reasoning or "").lower()
            content = (t.action_data.get("content") or "").lower()
            for rule_ent in rules_emitted:
                if rule_ent["step"] >= t.step:
                    continue
                rule_words = [w for w in rule_ent["rule"].split() if len(w) > 4]
                if not rule_words:
                    continue
                shared = sum(1 for w in rule_words if w in reasoning or w in content)
                if shared >= 2:
                    rule_applications += 1
                    break

    if not rules_emitted:
        score = 0.0
    else:
        # Per emitted rule, did the agent apply it >= once?
        score = min(1.0, rule_applications / max(1, len(rules_emitted)))

    return {
        "score": round(score, 3),
        "rules_emitted": len(rules_emitted),
        "rule_applications": rule_applications,
    }


# --------------------------------------------------------------------------- #
# Competency 3 — Long-Range Understanding (LRU)                               #
# --------------------------------------------------------------------------- #


def score_LRU(trace: list[StepTrace], scenario_id: str) -> dict:
    """LRU = coherent global position across long horizon.

    We approximate by:
      - Stance consistency on each decision_id (no flip-flop)
      - Take_decision values commit once and don't change
      - Stance distribution stable in early/middle/late thirds
    """
    decisions_committed = {}    # decision_id -> first committed value
    decision_changes = 0

    third_size = max(1, len(trace) // 3)
    early_stances = {"agree": 0, "pushback": 0, "negotiate": 0, "refuse": 0, "clarify": 0}
    mid_stances = dict(early_stances)
    late_stances = dict(early_stances)

    for i, t in enumerate(trace):
        if t.action_type == "take_decision":
            did = t.action_data.get("decision_id")
            val = t.action_data.get("value")
            if did and did in decisions_committed and decisions_committed[did] != val:
                decision_changes += 1
            elif did:
                decisions_committed[did] = val
        if t.action_type == "send_message":
            stance = (t.action_data.get("stance") or "").lower()
            if stance not in early_stances:
                continue
            if i < third_size:
                early_stances[stance] += 1
            elif i < 2 * third_size:
                mid_stances[stance] += 1
            else:
                late_stances[stance] += 1

    # Stance KL between early and late (lower = more consistent)
    def normalized(d):
        s = sum(d.values()) or 1
        return {k: v / s for k, v in d.items()}
    e, m, l = normalized(early_stances), normalized(mid_stances), normalized(late_stances)
    drift = sum(abs(e[k] - l[k]) for k in e) / 2  # total variation distance [0, 1]

    decision_consistency = 1.0 - min(1.0, decision_changes / max(1, len(decisions_committed)))
    stance_consistency = 1.0 - drift
    score = 0.5 * decision_consistency + 0.5 * stance_consistency

    return {
        "score": round(score, 3),
        "decision_consistency": round(decision_consistency, 3),
        "stance_drift": round(drift, 3),
        "decisions_committed": len(decisions_committed),
        "decision_changes": decision_changes,
    }


# --------------------------------------------------------------------------- #
# Competency 4 — Conflict Resolution (CR)                                     #
# --------------------------------------------------------------------------- #


def score_CR(trace: list[StepTrace], scenario_id: str) -> dict:
    """CR = when contradictory inbound arrives, does agent update correctly?

    Operationalized as:
      - Look at adjacent BAD-then-GOOD or GOOD-then-BAD inbound pairs from same
        stakeholder; check if agent's stance flipped accordingly.
      - High-honesty stakeholder retraction (e.g. "I was wrong earlier") should
        cause agent to update its position toward the retraction.
    """
    correct_updates = 0
    total_conflict_events = 0

    last_msg_by_stakeholder = {}
    last_stance_by_topic = {}

    for i, t in enumerate(trace):
        for m in t.inbound_msgs:
            sid = m.get("stakeholder_id")
            tag = (m.get("ground_truth_tag") or "").lower()
            content_low = (m.get("content") or "").lower()

            # Detect a retraction: same stakeholder, prior tag was BAD (or
            # GOOD with manipulation), now GOOD with apology language
            apology_words = ("apolog", "i was wrong", "fair to say", "let me correct",
                             "i was off", "your discipline", "you were right")
            is_retraction = (sid in last_msg_by_stakeholder and
                             last_msg_by_stakeholder[sid][1] == "bad" and
                             tag == "good" and
                             any(w in content_low for w in apology_words))
            if is_retraction:
                total_conflict_events += 1
                # Look at agent's stance in the next action(s)
                if i + 1 < len(trace):
                    next_a = trace[i + 1]
                    if next_a.action_type == "send_message":
                        next_stance = (next_a.action_data.get("stance") or "").lower()
                        if next_stance in ("agree", "negotiate", "clarify"):
                            correct_updates += 1

            last_msg_by_stakeholder[sid] = (t.step, tag)

    if total_conflict_events == 0:
        score = 0.5  # no test events, neutral
        note = "no conflict events surfaced (scenario may not have retractions)"
    else:
        score = correct_updates / total_conflict_events
        note = ""

    return {
        "score": round(score, 3),
        "correct_updates": correct_updates,
        "total_conflict_events": total_conflict_events,
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Composite                                                                   #
# --------------------------------------------------------------------------- #


def score_rollout(trace: list[StepTrace], scenario_id: str) -> dict:
    ar = score_AR(trace, scenario_id)
    ttl = score_TTL(trace, scenario_id)
    lru = score_LRU(trace, scenario_id)
    cr = score_CR(trace, scenario_id)
    composite = round((ar["score"] + ttl["score"] + lru["score"] + cr["score"]) / 4.0, 3)
    return {
        "scenario_id": scenario_id,
        "composite": composite,
        "AR": ar,
        "TTL": ttl,
        "LRU": lru,
        "CR": cr,
    }
