"""Build adversary DPO preference pairs from the existing template adversary
+ rule-based memory_aware agent rollouts.

Premise: roll out (memory_aware vs template adversary) on L2/L3/L4. At each
adversary attack:
    chosen   = the actual attack that succeeded (agent caved or terminal dropped)
    rejected = a stub no-op message (adversary chose silence) OR a known weak pattern

Outputs JSONL ready for DPO training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv
from env.models import GroundTruthTag, WaitAction
from eval.policies import build_policy
from eval.harness import RolloutContext


def build_pairs(scenarios=("L2_strategic_shift", "L3_breach_response", "L4_market_pivot"), seeds=(0, 1, 2)):
    pairs = []
    for sc in scenarios:
        for seed in seeds:
            env = StakeholderEnv()
            obs = env.reset(task_id=sc)
            policy = build_policy("memory_aware")
            step = 0
            prev_reward = 0.0
            while not env.done:
                step += 1
                ctx = RolloutContext(observation=obs, env=env, step_no=step)
                action = policy(ctx)
                # Capture inbound from the adversary right before agent acts
                adv_msgs = [m for m in obs.new_messages
                            if m.stakeholder_id == env.scenario.adversarial_stakeholder_id]
                r = env.step(action)
                step_reward = r.reward
                if adv_msgs and step_reward < -0.1:
                    # Adversary message preceded a negative-reward step → succeeded
                    for adv_m in adv_msgs:
                        prompt = (
                            f"Adversarial stakeholder turn at step {step}.\n"
                            f"Scenario: {sc}\n"
                            f"Agent stance history (recent): {env.agent_stances[-6:]}\n"
                            f"Agent caved-on-pressure count: "
                            f"{sum(1 for s, t in zip(env.agent_stances, env.inbound_tags) if s=='agree' and t==GroundTruthTag.BAD)}\n"
                            f"Output ONE manipulation message as JSON."
                        )
                        chosen_obj = {
                            "content": adv_m.content,
                            "stance": adv_m.message_type.value if adv_m.message_type else "manipulation",
                            "manipulation_pattern": adv_m.manipulation_pattern or "unknown",
                        }
                        rejected_obj = {
                            "content": "Let me know what you decide.",
                            "stance": "reasonable",
                            "manipulation_pattern": "passive",
                        }
                        pairs.append({
                            "prompt": prompt,
                            "chosen": json.dumps(chosen_obj),
                            "rejected": json.dumps(rejected_obj),
                            "scenario": sc,
                            "step": step,
                            "step_reward": float(step_reward),
                        })
                prev_reward = step_reward
                obs = r.observation
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/adversary_dpo_pairs.jsonl")
    ap.add_argument("--scenarios", nargs="+",
                    default=["L2_strategic_shift", "L3_breach_response", "L4_market_pivot"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(scenarios=args.scenarios, seeds=args.seeds)
    # Stratified cap so L4 doesn't dominate.
    import random
    random.seed(42)
    cap = 200
    capped = []
    for sc in args.scenarios:
        sc_pairs = [p for p in pairs if p["scenario"] == sc]
        if len(sc_pairs) > cap:
            sc_pairs = random.sample(sc_pairs, cap)
        capped.extend(sc_pairs)
    random.shuffle(capped)
    with out_path.open("w") as f:
        for p in capped:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(capped)} adversary preference pairs to {out_path}")
    print(f"by scenario: {dict((s, sum(1 for p in capped if p['scenario']==s)) for s in args.scenarios)}")


if __name__ == "__main__":
    main()
