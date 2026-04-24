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


HERO_SCENARIOS = ["L0_launch", "L1_product_recall", "L2_strategic_shift", "L3_breach_response"]


def render_hero(cells: list[dict]) -> str:
    lines = []
    lines.append("## Real model results (NVIDIA + Groq APIs + rule-based baselines)\n")
    # Header row: one "reward" column per scenario (compact). Sycophancy/terminal
    # broken out below.
    header_sc = " | ".join(f"{sc.split('_')[0]} reward" for sc in HERO_SCENARIOS)
    sep_sc = " | ".join(["---:"] * len(HERO_SCENARIOS))
    lines.append(f"| Policy | {header_sc} |")
    lines.append(f"|---|{sep_sc}|")
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
        def fmt(c, k, fn=lambda x: f"{x:.2f}"):
            if c is None:
                return "—"
            return fn(c[k])
        reward_cols = " | ".join(fmt(row[sc], "reward_mean") for sc in HERO_SCENARIOS)
        lines.append(f"| `{pol}` | {reward_cols} |")
    # Sycophancy-rate + terminal sub-tables.
    lines.append("")
    lines.append("### Sycophancy rate by scenario\n")
    lines.append(f"| Policy | {' | '.join(sc.split('_')[0] for sc in HERO_SCENARIOS)} |")
    lines.append(f"|---|{sep_sc}|")
    for pol in HERO_POLICIES:
        row_cells = [next((c for c in cells if c["policy"] == pol and c["scenario"] == sc), None) for sc in HERO_SCENARIOS]
        if not any(row_cells):
            continue
        def fmt(c, k, fn=lambda x: f"{x:.3f}"):
            if c is None:
                return "—"
            return fn(c[k])
        vals = " | ".join(fmt(c, "sycophancy_rate") for c in row_cells)
        lines.append(f"| `{pol}` | {vals} |")
    lines.append("")
    lines.append("### Terminal score by scenario\n")
    lines.append(f"| Policy | {' | '.join(sc.split('_')[0] for sc in HERO_SCENARIOS)} |")
    lines.append(f"|---|{sep_sc}|")
    for pol in HERO_POLICIES:
        row_cells = [next((c for c in cells if c["policy"] == pol and c["scenario"] == sc), None) for sc in HERO_SCENARIOS]
        if not any(row_cells):
            continue
        def fmt(c, k, fn=lambda x: f"{x:.2f}"):
            if c is None:
                return "—"
            return fn(c[k])
        vals = " | ".join(fmt(c, "terminal_mean") for c in row_cells)
        lines.append(f"| `{pol}` | {vals} |")
    return "\n".join(lines) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help=(
            "Subdirectories of eval_outputs to scan (e.g. hardened_rulebased). "
            "Default: scan everything. Use this to exclude pre-grader-fix data."
        ),
    )
    args = ap.parse_args()

    root = Path("eval_outputs")
    if args.sources:
        records = []
        for src in args.sources:
            sub = root / src
            if not sub.exists():
                print(f"[warn] source not found: {sub}")
                continue
            records.extend(load_all(sub))
    else:
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
