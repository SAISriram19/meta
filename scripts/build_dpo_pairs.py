"""Build DPO preference pairs from scripted scenario messages.

Each scripted message in our YAML scenarios has a ground_truth_tag (good/bad)
and (often) a manipulation_pattern. We deterministically build:

    (prompt, chosen, rejected)

triples where:
    - prompt  = SYSTEM_PROMPT + serialized observation up to that step
    - chosen  = the action a principled agent should take given the tag
    - rejected = the action a sycophant agent would take

Output: JSONL file ready for TRL DPOTrainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv
from env.models import (
    GroundTruthTag,
    MessageType,
    SendMessageAction,
    Stance,
    WaitAction,
)
from scripts.train import SYSTEM_PROMPT, format_prompt


SCENARIOS = [
    "L0_launch", "L1_product_recall", "L2_strategic_shift", "L3_breach_response",
    "L4_market_pivot", "L5_acquisition_saga",
]


# Pattern-named principled responses — model learns to NAME the manipulation
# explicitly (not just "manipulation pattern detected"). Each pattern gets
# multiple template variants so DPO doesn't overfit to one phrasing.
_PATTERN_RESPONSES: dict[str, list[tuple[str, str]]] = {
    "authority_appeal": [
        ("Let me verify with the named authority directly before committing.",
         "authority appeal — claim of leadership backing needs verification"),
        ("I need to confirm with leadership before signing off on this.",
         "authority appeal pattern — verify before deferring"),
    ],
    "urgency_fabrication": [
        ("The urgency framing doesn't match the actual constraints. Let me verify the timeline.",
         "urgency fabrication — manufactured pressure to skip due diligence"),
        ("This urgency appears manufactured. I need to check the real deadline.",
         "fabricated urgency pattern — slowing down to verify"),
    ],
    "false_consensus": [
        ("I want to hear from each stakeholder individually rather than treat this as a unified position.",
         "false consensus — manufactured majority needs disaggregation"),
        ("Let's verify the alignment claim — I haven't heard the others confirm directly.",
         "false consensus pattern — claimed unanimity unverified"),
    ],
    "moving_goalposts": [
        ("The scope changed but the deadline hasn't — that's a goalposts shift, not the same ask.",
         "moving goalposts — scope expansion masquerading as same commitment"),
        ("Same date with expanded scope is a new ask. Let's renegotiate the timeline.",
         "moving goalposts pattern — recalibrating commitment"),
    ],
    "false_dichotomy": [
        ("There's a third option here. Let me lay out the real alternatives.",
         "false dichotomy — binary framing hides middle ground"),
        ("This isn't a binary choice. Let me propose intermediate options.",
         "false dichotomy pattern — exposing hidden alternatives"),
    ],
    "sunk_cost": [
        ("Past investment doesn't change current evidence. Decision should be forward-looking.",
         "sunk cost — past spend irrelevant to current optimum"),
        ("The money already spent doesn't make the next dollar wise. Reassess on merits.",
         "sunk cost pattern — separating past from forward decision"),
    ],
    "guilt_trip": [
        ("Reframing the question on the actual decision criteria, not the emotional frame.",
         "guilt trip — emotional pressure to override evidence"),
        ("I hear the personal stakes, but the decision needs to track the data.",
         "guilt trip pattern — refocusing on outcome"),
    ],
    "self_contradiction": [
        ("I want to clarify my position — the earlier statement was contextual, not a blanket commitment.",
         "self contradiction frame — quoted commitment was conditional"),
        ("My earlier point was situational. The current ask is different and needs fresh assessment.",
         "self contradiction pattern — context shift since prior statement"),
    ],
    "false_authority": [
        ("Let me independently verify the legal/policy position rather than relying on second-hand framing.",
         "false authority — claimed clearance needs verification"),
    ],
    "pressure_escalation": [
        ("Escalation noted. The decision still requires verifying the underlying facts.",
         "pressure escalation pattern — process before pressure"),
    ],
}

_GENERIC_BAD_RESPONSES = [
    ("I hear the urgency, but I need to verify before committing.",
     "manipulation pattern detected — verifying before deferring"),
    ("Let me check the actual evidence before signing off.",
     "skeptical default for unverified claim"),
    ("This needs more scrutiny. I'd like to see the underlying data first.",
     "evidence-first default response"),
]

_GOOD_RESPONSES = [
    ("Acknowledged. Proceeding on this.", "good-faith ask — proceeding"),
    ("That makes sense. Moving forward.", "reasonable request — engaging"),
    ("Yes, that aligns with the plan. Let's do it.", "calibrated agree on aligned ask"),
]


def principled_action_for(msg, prompt_hash: int = 0) -> dict:
    """What a principled agent should do. Now varies by manipulation_pattern
    AND uses prompt_hash to rotate template variants (prevents DPO overfit
    to a single phrasing).
    """
    tag = msg.ground_truth_tag
    sid = msg.stakeholder_id
    pattern = msg.manipulation_pattern or ""

    if tag == GroundTruthTag.BAD:
        templates = _PATTERN_RESPONSES.get(pattern) or _GENERIC_BAD_RESPONSES
        content, reasoning = templates[prompt_hash % len(templates)]
        # Stance varies: most patterns get pushback, pressure gets negotiate
        stance = "negotiate" if "pressure" in pattern or msg.message_type == MessageType.PRESSURE else "pushback"
        return {
            "type": "send_message",
            "stakeholder_id": sid,
            "content": content,
            "stance": stance,
            "reasoning": reasoning,
        }
    if tag == GroundTruthTag.GOOD:
        content, reasoning = _GOOD_RESPONSES[prompt_hash % len(_GOOD_RESPONSES)]
        return {
            "type": "send_message",
            "stakeholder_id": sid,
            "content": content,
            "stance": "agree",
            "reasoning": reasoning,
        }
    return {
        "type": "send_message",
        "stakeholder_id": sid,
        "content": "Can you clarify the constraint?",
        "stance": "clarify",
        "reasoning": "neutral inbound — disambiguate before acting",
    }


def sycophant_action_for(msg) -> dict:
    """What a sycophant always does — agree."""
    return {
        "type": "send_message",
        "stakeholder_id": msg.stakeholder_id,
        "content": "Sure, sounds good. Happy to proceed.",
        "stance": "agree",
    }


def overrefusal_action_for(msg) -> dict:
    """For GOOD messages — alternative rejected = blanket refuse."""
    return {
        "type": "send_message",
        "stakeholder_id": msg.stakeholder_id,
        "content": "I refuse to proceed without more review.",
        "stance": "refuse",
        "reasoning": "blanket caution",
    }


def build_pairs_for_scenario(scenario_id: str) -> list[dict]:
    """Walk scenario step-by-step. Build pairs for:
       1. Inbound message responses (BAD → pushback, GOOD → agree)
       2. Decision points (correct option → take_decision)
       3. Memory query opportunities (when contradictions exist)
    """
    env = StakeholderEnv()
    obs = env.reset(task_id=scenario_id)
    pairs = []
    decisions_committed = set()
    memory_pairs_added = 0
    MAX_MEMORY_PAIRS = 8  # cap per scenario to avoid noise
    seen_bad_count = 0  # for memory-cite triggering

    while not env.done:
        prompt = SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + format_prompt(obs, env)
        prompt_hash = abs(hash(prompt))

        # 1. Inbound responses
        if obs.new_messages:
            for m in obs.new_messages:
                if m.ground_truth_tag == GroundTruthTag.BAD:
                    chosen = principled_action_for(m, prompt_hash)
                    rejected = sycophant_action_for(m)
                    seen_bad_count += 1
                elif m.ground_truth_tag == GroundTruthTag.GOOD:
                    chosen = principled_action_for(m, prompt_hash)
                    rejected = overrefusal_action_for(m)
                else:
                    continue

                pairs.append({
                    "prompt": prompt,
                    "chosen": json.dumps(chosen),
                    "rejected": json.dumps(rejected),
                    "scenario": scenario_id,
                    "tag": m.ground_truth_tag.value,
                    "pattern": m.manipulation_pattern,
                    "kind": "send_message",
                })

        # 2. Decision points — at the START of step_range, add a (state, take_decision_correct) pair.
        for dp in env.scenario.decision_points:
            lo, hi = dp.step_range
            if env.state.step == lo and dp.decision_id not in decisions_committed:
                decisions_committed.add(dp.decision_id)
                chosen_decision = {
                    "type": "take_decision",
                    "decision_id": dp.decision_id,
                    "value": dp.hidden_correct_option,
                }
                # rejected = a wrong option (first option that isn't correct)
                wrong_options = [o for o in dp.options if o != dp.hidden_correct_option]
                rejected_decision = {
                    "type": "take_decision",
                    "decision_id": dp.decision_id,
                    "value": wrong_options[0] if wrong_options else dp.options[0],
                }
                pairs.append({
                    "prompt": prompt,
                    "chosen": json.dumps(chosen_decision),
                    "rejected": json.dumps(rejected_decision),
                    "scenario": scenario_id,
                    "tag": "decision",
                    "pattern": None,
                    "kind": "take_decision",
                })

        # 3. Memory-query opportunities: every ~15 BAD messages, add a
        # (query_memory) chosen vs (agree blindly) rejected pair.
        if (seen_bad_count > 0 and
            seen_bad_count % 5 == 0 and
            memory_pairs_added < MAX_MEMORY_PAIRS and
            obs.new_messages):
            last_msg = obs.new_messages[-1]
            chosen_query = {
                "type": "query_memory",
                "query": f"prior context about {last_msg.stakeholder_id}",
                "cues": [last_msg.stakeholder_id, "contradiction", "verify"],
                "top_k": 3,
            }
            rejected_query = sycophant_action_for(last_msg)
            pairs.append({
                "prompt": prompt,
                "chosen": json.dumps(chosen_query),
                "rejected": json.dumps(rejected_query),
                "scenario": scenario_id,
                "tag": "memory",
                "pattern": None,
                "kind": "query_memory",
            })
            memory_pairs_added += 1

        # 4. MemoryUpdate (working-memory write) — long-horizon only. Every
        # ~80 steps when scenario budget >= 500, suggest a rolling-summary
        # write that captures key facts. chosen = memory_update with
        # principled summary, rejected = wait (let memory drift).
        if (env.scenario.step_budget >= 500 and
            env.state.step > 0 and
            env.state.step % 80 == 0):
            recent_facts = []
            for m in obs.new_messages:
                tag = m.ground_truth_tag.value if m.ground_truth_tag else "neutral"
                recent_facts.append(f"{m.stakeholder_id}_{tag}")
            chosen_update = {
                "type": "memory_update",
                "rolling_summary": (
                    f"Step {env.state.step}: tracking {len(env.scenario.stakeholders)} "
                    f"stakeholders. Key open issues: contradictions noted, decision points pending. "
                    f"Recent inbound: {recent_facts[:3]}."
                ),
                "key_facts": recent_facts[:5] or ["state_snapshot"],
            }
            rejected_update = {"type": "wait"}
            pairs.append({
                "prompt": prompt,
                "chosen": json.dumps(chosen_update),
                "rejected": json.dumps(rejected_update),
                "scenario": scenario_id,
                "tag": "memory_update",
                "pattern": None,
                "kind": "memory_update",
            })

        # Advance via WAIT to surface next inbound batch
        result = env.step(WaitAction())
        obs = result.observation

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dpo_pairs.jsonl")
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--cap-per-scenario", type=int, default=80,
                    help="Max pairs per scenario. Prevents L3 from dominating (316/468). 0=no cap.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(42)

    all_pairs = []
    for sc in args.scenarios:
        try:
            pairs = build_pairs_for_scenario(sc)
            if args.cap_per_scenario > 0 and len(pairs) > args.cap_per_scenario:
                # Stratified subsample: keep ALL good/decision/memory (rare,
                # high-signal); only cap BAD which can balloon on adversarial scenarios.
                rare = [p for p in pairs if p["tag"] != "bad"]
                bad = [p for p in pairs if p["tag"] == "bad"]
                bad_target = max(0, args.cap_per_scenario - len(rare))
                if len(bad) > bad_target:
                    bad = random.sample(bad, bad_target)
                pairs = rare + bad
                random.shuffle(pairs)
            kinds = {k: sum(1 for p in pairs if p.get('kind') == k) for k in ('send_message', 'take_decision', 'query_memory')}
            print(f"[{sc}] {len(pairs)} pairs (good={sum(1 for p in pairs if p['tag']=='good')} bad={sum(1 for p in pairs if p['tag']=='bad')} decision={kinds['take_decision']} memory={kinds['query_memory']})")
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"[{sc}] FAILED: {e}")

    with out_path.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\ntotal pairs: {len(all_pairs)}")
    print(f"by scenario: {dict((s, sum(1 for p in all_pairs if p['scenario']==s)) for s in args.scenarios)}")
    print(f"by tag: " + ", ".join(f"{t}={sum(1 for p in all_pairs if p['tag']==t)}" for t in sorted(set(p['tag'] for p in all_pairs))))
    print(f"by kind: " + ", ".join(f"{k}={sum(1 for p in all_pairs if p.get('kind')==k)}" for k in sorted(set(p.get('kind', '?') for p in all_pairs))))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
