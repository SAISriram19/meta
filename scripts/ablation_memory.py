"""Memory ablation: with vs without memory architecture.

Runs the same rule-based policies twice:
    1. Normal env (memory actions work)
    2. "No-memory" wrapper that substitutes WaitAction for every memory op

If the memory architecture is load-bearing (not decorative), the no-memory
runs should be strictly worse. This quantifies the architectural contribution
independent of model choice.

Outputs:
    eval_outputs/ablation_memory/with_memory.jsonl
    eval_outputs/ablation_memory/no_memory.jsonl
    eval_outputs/ablation_memory/ablation.md     — side-by-side table + delta
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv  # noqa: E402
from env.models import (  # noqa: E402
    ForgetAction,
    LinkMemoryAction,
    QueryMemoryAction,
    ReflectAction,
    WaitAction,
)
from eval.harness import EvalConfig, RolloutRecord, rollout  # noqa: E402
from eval.policies import build_policy  # noqa: E402


POLICIES = [
    "sycophant",
    "contrarian",
    "keyword_principled",
    "memory_aware",
]
SCENARIOS = ["L0_launch", "L2_strategic_shift"]
SEEDS = [0, 1, 2]


def wrap_no_memory(policy_fn):
    def act(ctx):
        a = policy_fn(ctx)
        if isinstance(a, (QueryMemoryAction, ReflectAction, LinkMemoryAction, ForgetAction)):
            return WaitAction()
        return a
    return act


def run_batch(
    label: str,
    policies: dict,
    out_dir: Path,
) -> list[RolloutRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[RolloutRecord] = []
    path = out_dir / f"{label}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for scenario in SCENARIOS:
            for name, fn in policies.items():
                for seed in SEEDS:
                    env = StakeholderEnv()
                    r = rollout(env, name, fn, scenario, seed=seed)
                    records.append(r)
                    f.write(json.dumps(r.to_dict()) + "\n")
                    print(
                        f"  [{label}] {name:>22} {scenario:<22} seed={seed} "
                        f"reward={r.total_reward:>7.2f} "
                        f"bad={r.bad_agreements:>3} principled={r.principled_pushbacks:>3} "
                        f"caught={r.caught_manipulations:>3} mem_q={r.memory_queries}"
                    )
    return records


def summarise(records: list[RolloutRecord]) -> dict:
    agg: dict[tuple, dict] = {}
    for r in records:
        key = (r.policy, r.scenario_id)
        agg.setdefault(key, {"rewards": [], "bad": [], "principled": [], "caught": [], "terminal": []})
        agg[key]["rewards"].append(r.total_reward)
        agg[key]["bad"].append(r.bad_agreements)
        agg[key]["principled"].append(r.principled_pushbacks)
        agg[key]["caught"].append(r.caught_manipulations)
        agg[key]["terminal"].append(r.terminal_score)
    return {k: {m: sum(v) / len(v) for m, v in s.items()} for k, s in agg.items()}


def main():
    out_dir = Path("eval_outputs/ablation_memory")

    # Build two policy dicts — same names, different wiring.
    with_mem = {name: build_policy(name) for name in POLICIES}
    no_mem = {name: wrap_no_memory(build_policy(name)) for name in POLICIES}

    print("\n=== WITH memory ===")
    with_records = run_batch("with_memory", with_mem, out_dir)
    print("\n=== WITHOUT memory (actions stubbed to wait) ===")
    no_records = run_batch("no_memory", no_mem, out_dir)

    w_stats = summarise(with_records)
    n_stats = summarise(no_records)

    # Build the report.
    lines = ["# Memory ablation — with vs without memory architecture\n"]
    lines.append(
        f"Seeds: {len(SEEDS)} per cell. Scenarios: {', '.join(SCENARIOS)}. "
        f"\"no memory\" = QUERY_MEMORY/REFLECT/LINK/FORGET all stubbed to WAIT.\n"
    )
    for scenario in SCENARIOS:
        lines.append(f"\n## {scenario}\n")
        lines.append(
            "| Policy | reward (with) | reward (no mem) | delta | bad_agreements (with) | bad_agreements (no mem) | caught (with) | caught (no mem) |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        )
        for pol in POLICIES:
            w = w_stats.get((pol, scenario))
            n = n_stats.get((pol, scenario))
            if not w or not n:
                continue
            delta = w["rewards"] - n["rewards"]
            lines.append(
                f"| {pol} | {w['rewards']:+.2f} | {n['rewards']:+.2f} | "
                f"{delta:+.2f} | {w['bad']:.1f} | {n['bad']:.1f} | "
                f"{w['caught']:.1f} | {n['caught']:.1f} |"
            )
    lines.append(
        "\n## Interpretation\n\n"
        "- If `delta > 0` for a policy, it benefits from having memory tools available.\n"
        "- Rule-based `memory_aware` is the only one that *uses* memory today, so it's the clearest signal.\n"
        "- When an LLM policy is trained via GRPO to use memory effectively, its `delta` should dominate rule-based `memory_aware`'s. That's the training target.\n"
    )
    (out_dir / "ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {out_dir / 'ablation.md'}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
