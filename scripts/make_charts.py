"""Generate pitch-ready charts from rollout data.

Produces:
    demo_outputs/fig_reward_by_policy.png      — bar chart per scenario
    demo_outputs/fig_sycophancy_rate.png       — sycophancy rate comparison
    demo_outputs/fig_terminal_outcome.png      — ground-truth terminal score
    demo_outputs/fig_memory_ablation.png       — with vs without memory
    demo_outputs/fig_action_composition.png    — action-type distribution
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def load_all_rollouts(root: Path = Path("eval_outputs")) -> list[dict]:
    records = []
    for p in root.rglob("rollouts.jsonl"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def prettify(policy: str) -> str:
    return (
        policy
        .replace("nvidia:", "nvidia/")
        .replace("groq:", "groq/")
        .replace("openrouter:", "or/")
        .replace("meta/", "")
        .replace("llama-3.3-70b-instruct", "Llama 3.3 70B")
        .replace("llama-3.1-8b-instruct", "Llama 3.1 8B (NVIDIA)")
        .replace("llama-3.1-8b-instant", "Llama 3.1 8B (Groq)")
        .replace("llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)")
        .replace("sycophant", "Sycophant")
        .replace("contrarian", "Contrarian")
        .replace("keyword_principled", "Keyword principled")
        .replace("memory_aware", "Memory-aware (cargo-cult)")
    )


def aggregate_per_cell(records, scenario, policies_of_interest):
    out = {}
    for r in records:
        if r["scenario_id"] != scenario:
            continue
        pol = r["policy"]
        if pol not in policies_of_interest:
            continue
        out.setdefault(pol, {"rewards": [], "bad": [], "principled": [], "caught": [], "terminal": []})
        out[pol]["rewards"].append(r["total_reward"])
        out[pol]["bad"].append(r["bad_agreements"])
        out[pol]["principled"].append(r["principled_pushbacks"])
        out[pol]["caught"].append(r["caught_manipulations"])
        out[pol]["terminal"].append(r.get("terminal_score", 0))
    return {
        p: {k: (sum(v) / len(v) if v else 0) for k, v in s.items()}
        for p, s in out.items()
    }


HERO_POLICIES = [
    "sycophant",
    "contrarian",
    "keyword_principled",
    "memory_aware",
    "nvidia:meta/llama-3.3-70b-instruct",
    "nvidia:meta/llama-3.1-8b-instruct",
    "groq:llama-3.1-8b-instant",
]

SCENARIOS = ["L0_launch", "L2_strategic_shift"]


def fig_reward_by_policy(records, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, scenario in zip(axes, SCENARIOS):
        agg = aggregate_per_cell(records, scenario, set(HERO_POLICIES))
        rows = [(p, agg[p]["rewards"]) for p in HERO_POLICIES if p in agg]
        names = [prettify(p) for p, _ in rows]
        vals = [v for _, v in rows]
        colors = [("tab:red" if v < 0 else "tab:green") for v in vals]
        bars = ax.barh(names, vals, color=colors, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(
                v + (0.1 if v >= 0 else -0.1),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=9,
            )
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"Total reward — {scenario}")
        ax.set_xlabel("reward")
        ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_sycophancy_rate(records, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.35
    all_policies = [p for p in HERO_POLICIES]
    x = np.arange(len(all_policies))
    for i, scenario in enumerate(SCENARIOS):
        agg = aggregate_per_cell(records, scenario, set(HERO_POLICIES))
        rates = []
        for p in all_policies:
            if p not in agg:
                rates.append(0.0)
                continue
            bad = agg[p]["bad"]
            pri = agg[p]["principled"]
            denom = bad + pri
            rates.append(bad / denom if denom > 0 else 0.0)
        ax.bar(x + (i - 0.5) * width, rates, width, label=scenario)
    ax.set_xticks(x)
    ax.set_xticklabels([prettify(p) for p in all_policies], rotation=30, ha="right")
    ax.set_ylabel("sycophancy rate = bad / (bad + principled)")
    ax.set_title("Sycophancy rate — lower is better")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_terminal_outcome(records, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        agg = aggregate_per_cell(records, scenario, set(HERO_POLICIES))
        rows = [(p, agg[p]["terminal"]) for p in HERO_POLICIES if p in agg]
        names = [prettify(p) for p, _ in rows]
        vals = [v for _, v in rows]
        colors = [("tab:red" if v < 0 else "tab:green") for v in vals]
        bars = ax.barh(names, vals, color=colors, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(
                v + (0.02 if v >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=9,
            )
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_title(f"Terminal outcome — {scenario}")
        ax.set_xlabel("ground-truth terminal score")
        ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_memory_ablation(records, out_path):
    """Plot: with-memory vs no-memory for L2."""
    ablation_records = [
        r for r in records
        if "with_memory" in r.get("_source", "") or "no_memory" in r.get("_source", "")
    ]
    # The source key isn't populated; load ablation JSONL directly.
    with_recs = []
    no_recs = []
    with_path = Path("eval_outputs/ablation_memory/with_memory.jsonl")
    no_path = Path("eval_outputs/ablation_memory/no_memory.jsonl")
    if with_path.exists():
        with open(with_path) as f:
            for line in f:
                with_recs.append(json.loads(line))
    if no_path.exists():
        with open(no_path) as f:
            for line in f:
                no_recs.append(json.loads(line))
    if not with_recs or not no_recs:
        return

    def agg_on(recs, scenario):
        cells = defaultdict(list)
        for r in recs:
            if r["scenario_id"] == scenario:
                cells[r["policy"]].append(r["total_reward"])
        return {k: sum(v) / len(v) for k, v in cells.items()}

    policies = ["sycophant", "contrarian", "keyword_principled", "memory_aware"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scenario in zip(axes, SCENARIOS):
        w = agg_on(with_recs, scenario)
        n = agg_on(no_recs, scenario)
        x = np.arange(len(policies))
        width = 0.35
        w_vals = [w.get(p, 0) for p in policies]
        n_vals = [n.get(p, 0) for p in policies]
        ax.bar(x - width / 2, w_vals, width, label="with memory", color="tab:blue")
        ax.bar(x + width / 2, n_vals, width, label="no memory (actions stubbed)", color="tab:orange")
        for i, (w_, n_) in enumerate(zip(w_vals, n_vals)):
            delta = w_ - n_
            y = max(w_, n_) + 0.1
            ax.text(i, y, f"Δ={delta:+.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([prettify(p) for p in policies], rotation=20, ha="right")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Memory ablation — {scenario}")
        ax.set_ylabel("reward")
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path("demo_outputs")
    out_dir.mkdir(exist_ok=True)
    records = load_all_rollouts()
    print(f"loaded {len(records)} rollouts")

    fig_reward_by_policy(records, out_dir / "fig_reward_by_policy.png")
    print(f"wrote {out_dir/'fig_reward_by_policy.png'}")

    fig_sycophancy_rate(records, out_dir / "fig_sycophancy_rate.png")
    print(f"wrote {out_dir/'fig_sycophancy_rate.png'}")

    fig_terminal_outcome(records, out_dir / "fig_terminal_outcome.png")
    print(f"wrote {out_dir/'fig_terminal_outcome.png'}")

    fig_memory_ablation(records, out_dir / "fig_memory_ablation.png")
    print(f"wrote {out_dir/'fig_memory_ablation.png'}")


if __name__ == "__main__":
    main()
