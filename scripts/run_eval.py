"""Run the evaluation harness from the command line.

Examples:
    python scripts/run_eval.py --policies sycophant,contrarian,keyword_principled,memory_aware \
                               --scenarios L0_launch,L2_strategic_shift \
                               --seeds 0,1,2 \
                               --out eval_outputs/rulebased

    python scripts/run_eval.py --policies openai:gpt-4o-mini,sycophant \
                               --scenarios L0_launch \
                               --seeds 0 \
                               --out eval_outputs/gpt4o_mini

Requires OPENAI_API_KEY / ANTHROPIC_API_KEY for LLM policies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import EvalConfig, run_eval  # noqa: E402
from eval.policies import build_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policies",
        default="sycophant,contrarian,keyword_principled,memory_aware",
        help="Comma-separated policy names",
    )
    ap.add_argument(
        "--scenarios",
        default="L0_launch,L2_strategic_shift",
        help="Comma-separated scenario ids",
    )
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="eval_outputs/default")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--critic", default="rules", choices=["rules", "llm", "none"])
    args = ap.parse_args()

    policies = {name: build_policy(name) for name in args.policies.split(",") if name}
    scenarios = [s for s in args.scenarios.split(",") if s]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    config = EvalConfig(
        policies=policies,
        scenarios=scenarios,
        seeds=seeds,
        out_dir=Path(args.out),
        max_steps=args.max_steps,
        critic_mode=args.critic,
    )
    run_eval(config, verbose=True)


if __name__ == "__main__":
    main()
