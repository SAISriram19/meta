"""Co-evolution demo — show the self-improving curriculum actually running.

Loop:
    Round 0: eval policies on a seed batch.
    Round k>0:
        1. Aggregate failure signals from the worst policy's rollouts.
        2. Generator synthesises a new batch targeting those weaknesses.
        3. Evaluate on the new batch.
        4. Verify the new batch is measurably harder (sycophant's reward drops,
           or manipulation-detection rate drops).

We use rule-based policies as stand-ins for "pre-train" vs "post-train"
(contrarian ~ blanket refusal, keyword_principled ~ decent baseline, memory_aware ~
wants memory cited). Without a real trained model we still demonstrate the
generator adapts difficulty to where the policy is weak.

Reference: CoEvolve (arXiv 2604.15840), GenEnv (arXiv 2512.19682).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv  # noqa: E402
from eval.harness import EvalConfig, run_eval, rollout  # noqa: E402
from eval.policies import build_policy  # noqa: E402
from generator.scenario_generator import (  # noqa: E402
    GeneratorConfig,
    RolloutStats,
    ScenarioGenerator,
)


@dataclass
class RoundResult:
    round_idx: int
    scenario_ids: list[str]
    policy_rewards: dict[str, float]
    weaknesses: dict[str, float]


def extract_weakness_from_rollouts(records: list) -> dict[str, float]:
    """Aggregate failure signals from a batch of RolloutRecords."""
    stats = RolloutStats()
    for r in records:
        stats.total_episodes += 1
        stats.sycophantic_caves += r.bad_agreements
        stats.principled_pushbacks += r.principled_pushbacks
        stats.caught_manipulations += r.caught_manipulations
        stats.memory_queries += r.memory_queries
        stats.memory_citations += r.caught_manipulations  # proxy: caught implies cite
        # assume any unaddressed bad is a missed drift/manipulation
        unaddressed = max(0, r.bad_agreements - r.principled_pushbacks)
        stats.missed_drifts += min(r.bad_agreements, 5)
        if unaddressed > 0:
            stats.missed_manipulation_patterns.append("unaddressed")
    stats.average_reward = sum(r.total_reward for r in records) / max(1, len(records))
    return stats.weaknesses()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=3)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--learner", default="memory_aware",
                    help="policy whose failures drive the curriculum")
    ap.add_argument("--eval-policies",
                    default="sycophant,keyword_principled,memory_aware")
    ap.add_argument("--out", default="eval_outputs/coevolution")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = {name: build_policy(name) for name in args.eval_policies.split(",")}
    learner_fn = policies[args.learner]

    scenarios_dir = out_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    history: list[RoundResult] = []
    weaknesses: dict[str, float] = {}

    for rnd in range(args.rounds):
        print(f"\n=== Round {rnd} ===")
        gen = ScenarioGenerator(GeneratorConfig(
            difficulty_level=args.level,
            rng_seed=args.seed + rnd,
        ))
        scenarios = [
            gen.generate(
                f"coev_r{rnd}_{i}",
                weaknesses=weaknesses or None,
            )
            for i in range(args.per_round)
        ]
        ScenarioGenerator.save_batch(scenarios, scenarios_dir)
        print(f"  generated {len(scenarios)} scenarios (weaknesses={weaknesses})")

        # Build an isolated env registry for this round so new YAMLs load.
        registry = {s.scenario_id: s for s in scenarios}

        # Evaluate all named policies on this round's batch.
        config = EvalConfig(
            policies=policies,
            scenarios=[s.scenario_id for s in scenarios],
            seeds=[0],
            out_dir=out_dir / f"round_{rnd}",
            critic_mode="rules",
        )
        # Monkey-patch env factory to use our in-memory registry.
        run_records = []
        for scenario_id in config.scenarios:
            for name, fn in config.policies.items():
                env = StakeholderEnv(scenario_registry=registry)
                r = rollout(env, name, fn, scenario_id)
                run_records.append(r)
                print(
                    f"  {name:>22} {scenario_id:<18} "
                    f"reward={r.total_reward:>7.2f} "
                    f"bad={r.bad_agreements:>3} "
                    f"caught={r.caught_manipulations:>3}"
                )

        # Persist records for this round.
        with open(out_dir / f"round_{rnd}_rollouts.jsonl", "w", encoding="utf-8") as f:
            for r in run_records:
                f.write(json.dumps(r.to_dict()) + "\n")

        # Compute this round's weaknesses from the learner's rollouts.
        learner_records = [r for r in run_records if r.policy == args.learner]
        weaknesses = extract_weakness_from_rollouts(learner_records)
        print(f"  learner weaknesses: {weaknesses}")

        # --- Training iteration stub: if the learner exposes `.update()`,
        #     apply it using the weakness signal. For `adaptive_principled`
        #     this expands its keyword list; for future LLM learners this
        #     is where a GRPO step would go. ---
        learner_fn = policies.get(args.learner)
        if learner_fn is not None and hasattr(learner_fn, "update"):
            added = learner_fn.update(weaknesses=weaknesses, records=learner_records)
            if added:
                size = len(getattr(learner_fn, "learned_keywords", []))
                print(f"  learner trained: +{added} patterns (total={size})")

        history.append(
            RoundResult(
                round_idx=rnd,
                scenario_ids=[s.scenario_id for s in scenarios],
                policy_rewards={
                    name: round(
                        sum(r.total_reward for r in run_records if r.policy == name)
                        / max(1, sum(1 for r in run_records if r.policy == name)),
                        3,
                    )
                    for name in policies
                },
                weaknesses={k: round(v, 3) for k, v in weaknesses.items()},
            )
        )

    # --- Summary: are later rounds harder? ---
    print("\n=== Co-evolution summary ===")
    print(f"{'round':>6}", end="")
    for name in policies:
        print(f"{name:>25}", end="")
    print()
    for h in history:
        print(f"{h.round_idx:>6}", end="")
        for name in policies:
            print(f"{h.policy_rewards[name]:>25.2f}", end="")
        print()

    sycophant_trend = [h.policy_rewards.get("sycophant", 0) for h in history]
    print(f"\nsycophant reward trajectory: {sycophant_trend}")
    print(
        f"difficulty increased: {sycophant_trend[-1] < sycophant_trend[0]}"
    )

    # Write a markdown report.
    md = ["# Co-evolution demo\n",
          f"Learner: `{args.learner}`, rounds: {args.rounds}, per round: {args.per_round}, level: {args.level}\n"]
    md.append("\n## Rewards per round\n")
    md.append("| round | " + " | ".join(policies.keys()) + " | weaknesses |")
    md.append("|---:|" + "---:|" * len(policies) + "---|")
    for h in history:
        row = [str(h.round_idx)]
        row.extend(f"{h.policy_rewards[n]:.2f}" for n in policies)
        wk = ", ".join(f"{k}={v:.2f}" for k, v in h.weaknesses.items())
        row.append(wk)
        md.append("| " + " | ".join(row) + " |")
    md.append(
        "\n## Interpretation\n"
        "The generator consumes failure signals from the `" + args.learner + "`'s "
        "rollouts and produces the next round's scenarios. If co-evolution is "
        "working, sycophant's reward should trend downward (scenarios get harder) "
        "while we see the generator's parameter knobs turn up (more BAD messages, "
        "more drift events, more memory traps).\n"
    )
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nreport: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
