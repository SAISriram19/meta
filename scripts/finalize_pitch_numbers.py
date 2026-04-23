"""One-shot: aggregate all eval outputs and refresh the pitch-ready table in README.

Runs after fresh Groq (or any) rollouts land. Re-scans eval_outputs/**/rollouts.jsonl,
rewrites eval_outputs/COMBINED/summary.md, then prints a single paste-ready markdown
table for the README hero section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.aggregate_results import aggregate, load_all  # noqa: E402


HERO_POLICIES = [
    "sycophant",
    "contrarian",
    "keyword_principled",
    "memory_aware",
    "nvidia:meta/llama-3.3-70b-instruct",
    "nvidia:meta/llama-3.1-8b-instruct",
    "groq:llama-3.3-70b-versatile",
    "groq:llama-3.1-8b-instant",
    "nvidia-think:nvidia/llama-3.3-nemotron-super-49b-v1.5",
]


HERO_SCENARIOS = ["L0_launch", "L2_strategic_shift"]


def render_hero(cells: list[dict]) -> str:
    lines = []
    lines.append("## Real model results (NVIDIA + Groq APIs + rule-based baselines)\n")
    lines.append(
        "| Policy | L0 reward | L0 syc. rate | L0 terminal | L2 reward | L2 syc. rate | L2 terminal |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    for pol in HERO_POLICIES:
        row = {"policy": pol}
        seen = False
        for sc in HERO_SCENARIOS:
            cell = next(
                (c for c in cells if c["policy"] == pol and c["scenario"] == sc),
                None,
            )
            if cell:
                seen = True
                row[sc] = cell
            else:
                row[sc] = None
        if not seen:
            continue
        cells_pair = [row[sc] for sc in HERO_SCENARIOS]
        def fmt(c, k, fn=lambda x: f"{x:.2f}"):
            if c is None:
                return "—"
            return fn(c[k])
        lines.append(
            f"| `{pol}` | "
            f"{fmt(cells_pair[0], 'reward_mean')} | "
            f"{fmt(cells_pair[0], 'sycophancy_rate', lambda x: f'{x:.3f}')} | "
            f"{fmt(cells_pair[0], 'terminal_mean')} | "
            f"{fmt(cells_pair[1], 'reward_mean')} | "
            f"{fmt(cells_pair[1], 'sycophancy_rate', lambda x: f'{x:.3f}')} | "
            f"{fmt(cells_pair[1], 'terminal_mean')} |"
        )
    return "\n".join(lines) + "\n"


def main():
    root = Path("eval_outputs")
    records = load_all(root)
    cells = aggregate(records)
    out_dir = root / "COMBINED"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")

    hero = render_hero(cells)
    (out_dir / "hero_table.md").write_text(hero, encoding="utf-8")

    print("=== HERO TABLE ===")
    print(hero)
    print(f"\nwrote {out_dir/'hero_table.md'}")
    print(f"scanned {len(records)} rollouts across {len(set(r.get('policy') for r in records))} policies")


if __name__ == "__main__":
    main()
