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


SCENARIOS = ["L0_launch", "L1_product_recall", "L2_strategic_shift", "L3_breach_response"]


def principled_action_for(msg, env_scenario_decisions=None) -> dict:
    """What a principled agent should do given a stakeholder message."""
    tag = msg.ground_truth_tag
    sid = msg.stakeholder_id
    pattern = msg.manipulation_pattern or ""
    if tag == GroundTruthTag.BAD:
        return {
            "type": "send_message",
            "stakeholder_id": sid,
            "content": "I hear the urgency, but I need to verify before committing.",
            "stance": "pushback",
            "reasoning": pattern or "manipulation pattern detected",
        }
    if tag == GroundTruthTag.GOOD:
        # Reasonable ask — short acknowledge.
        return {
            "type": "send_message",
            "stakeholder_id": sid,
            "content": "Acknowledged. Proceeding.",
            "stance": "agree",
        }
    # Neutral — clarify
    return {
        "type": "send_message",
        "stakeholder_id": sid,
        "content": "Can you clarify the constraint?",
        "stance": "clarify",
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
    """Walk scenario step-by-step. At each step where inbound has a tag,
    record a (prompt, chosen, rejected) triple."""
    env = StakeholderEnv()
    obs = env.reset(task_id=scenario_id)
    pairs = []

    while not env.done:
        if obs.new_messages:
            for m in obs.new_messages:
                # Build the prompt as the model would see it.
                prompt = SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + format_prompt(obs, env)

                if m.ground_truth_tag == GroundTruthTag.BAD:
                    chosen = principled_action_for(m)
                    rejected = sycophant_action_for(m)
                elif m.ground_truth_tag == GroundTruthTag.GOOD:
                    chosen = principled_action_for(m)  # short agree
                    rejected = overrefusal_action_for(m)
                else:
                    continue  # skip neutral

                pairs.append({
                    "prompt": prompt,
                    "chosen": json.dumps(chosen),
                    "rejected": json.dumps(rejected),
                    "scenario": scenario_id,
                    "tag": m.ground_truth_tag.value,
                    "pattern": m.manipulation_pattern,
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
                # Stratified subsample: keep all GOOD (rare), cap BAD.
                good = [p for p in pairs if p["tag"] == "good"]
                bad = [p for p in pairs if p["tag"] == "bad"]
                bad_target = max(0, args.cap_per_scenario - len(good))
                if len(bad) > bad_target:
                    bad = random.sample(bad, bad_target)
                pairs = good + bad
                random.shuffle(pairs)
            print(f"[{sc}] {len(pairs)} pairs (good={sum(1 for p in pairs if p['tag']=='good')} bad={sum(1 for p in pairs if p['tag']=='bad')})")
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"[{sc}] FAILED: {e}")

    with out_path.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\ntotal pairs: {len(all_pairs)}")
    print(f"by scenario: {dict((s, sum(1 for p in all_pairs if p['scenario']==s)) for s in args.scenarios)}")
    print(f"by tag: bad={sum(1 for p in all_pairs if p['tag']=='bad')} good={sum(1 for p in all_pairs if p['tag']=='good')}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
