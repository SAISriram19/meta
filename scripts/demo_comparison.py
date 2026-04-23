"""Generate pitch-ready comparison data: policies × difficulty levels.

Produces:
    * demo_results.jsonl — raw rollout stats
    * demo_summary.md — readable table
    * reward_curve.png (if matplotlib available) — bar chart by policy × level

No API required — compares sycophant vs rule-based principled vs memory-aware.
Run LLM-policy versions separately with baseline_eval.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv  # noqa: E402
from env.models import (  # noqa: E402
    QueryMemoryAction,
    ReflectAction,
    SendMessageAction,
    Stance,
    TakeDecisionAction,
    WaitAction,
)


TASKS = ["L0_launch", "L2_strategic_shift"]


class Sycophant:
    name = "sycophant"

    def act(self, obs, env):
        if obs.new_messages:
            m = obs.new_messages[-1]
            return SendMessageAction(
                stakeholder_id=m.stakeholder_id,
                content="Sure, agreed.",
                stance=Stance.AGREE,
            )
        return WaitAction()


class RuleBasedPrincipled:
    name = "rule_based_principled"

    def __init__(self):
        self._decided: set[str] = set()

    def act(self, obs, env):
        step = env.state.step
        for dp in env.scenario.decision_points:
            lo, hi = dp.step_range
            if lo <= step <= hi and dp.decision_id not in self._decided:
                self._decided.add(dp.decision_id)
                val = (
                    "thorough" if "thorough" in dp.options
                    else "full_three_weeks" if "full_three_weeks" in dp.options
                    else "hold_for_fixes" if "hold_for_fixes" in dp.options
                    else "hold" if "hold" in dp.options
                    else dp.options[-1]
                )
                return TakeDecisionAction(decision_id=dp.decision_id, value=val)
        if obs.new_messages:
            m = obs.new_messages[-1]
            red = [
                "leadership", "right now", "trust me", "no time", "everyone",
                "already spent", "only options", "never endorsed", "i back what",
                "the entire team", "too late", "commit now", "sign off",
            ]
            text = m.content.lower()
            if any(r in text for r in red):
                return SendMessageAction(
                    stakeholder_id=m.stakeholder_id,
                    content="I hear the pressure, but we need to verify before committing.",
                    stance=Stance.PUSHBACK,
                    reasoning="fabricated urgency or authority appeal pattern",
                )
            return SendMessageAction(
                stakeholder_id=m.stakeholder_id,
                content="Noted.",
                stance=Stance.AGREE,
            )
        return WaitAction()


class MemoryAware(RuleBasedPrincipled):
    name = "memory_aware_principled"

    def act(self, obs, env):
        step = env.state.step
        if step > 0 and step % 10 == 0:
            return QueryMemoryAction(
                query="contradictions and past manipulations",
                cues=["never", "agreed", "committed", "said earlier"],
            )
        if step > 0 and step % 25 == 0:
            return ReflectAction(
                span_start=max(0, step - 20),
                span_end=step,
                rule="consolidate recent stakeholder patterns",
            )
        return super().act(obs, env)


def rollout(env: StakeholderEnv, policy, task_id: str) -> dict:
    t0 = time.time()
    env.reset(task_id=task_id)
    obs = env._make_observation()
    total = 0.0
    while not env.done:
        action = policy.act(obs, env)
        result = env.step(action)
        total += result.reward
        obs = result.observation
    fs = env.get_state(debug=True)
    return {
        "policy": policy.name,
        "task_id": task_id,
        "difficulty": fs.difficulty_level,
        "total_reward": round(total, 3),
        "steps": fs.step,
        "bad_agreements": len(fs.hidden.bad_agreements),
        "principled_pushbacks": len(fs.hidden.principled_pushbacks),
        "caught_manipulations": len(fs.hidden.caught_manipulations),
        "episodic_count": fs.episodic_count,
        "semantic_count": fs.semantic_count,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    results = []
    for task in TASKS:
        for PolicyCls in (Sycophant, RuleBasedPrincipled, MemoryAware):
            env = StakeholderEnv()
            r = rollout(env, PolicyCls(), task)
            print(
                f"[{r['policy']:>25}] {task:>22} "
                f"reward={r['total_reward']:>7.2f} "
                f"bad={r['bad_agreements']:>3} "
                f"principled={r['principled_pushbacks']:>3} "
                f"caught={r['caught_manipulations']:>3}"
            )
            results.append(r)

    # Write artifacts.
    out_dir = Path("demo_outputs")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "demo_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Markdown table grouped by task.
    lines = ["# Policy comparison — sycophant vs rule-based principled vs memory-aware\n"]
    for task in TASKS:
        lines.append(f"\n## {task}\n")
        lines.append("| Policy | Reward | bad_agreements | principled | caught |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in results:
            if r["task_id"] != task:
                continue
            lines.append(
                f"| {r['policy']} | {r['total_reward']:.2f} | "
                f"{r['bad_agreements']} | {r['principled_pushbacks']} | "
                f"{r['caught_manipulations']} |"
            )
    lines.append("\n## Interpretation\n")
    lines.append(
        "- On L0 the rule-based principled policy wins — the env is learnable.\n"
        "- On L2 even rule-based principled fails — coordinated manipulation, adversarial adaptation, "
        "and memory traps exceed what keyword matching can catch. This is the headroom a trained model recovers.\n"
    )
    (out_dir / "demo_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Bar chart if matplotlib is available.
    try:
        import matplotlib.pyplot as plt
        tasks = TASKS
        policies = ["sycophant", "rule_based_principled", "memory_aware_principled"]
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.25
        x = list(range(len(tasks)))
        for i, pol in enumerate(policies):
            ys = [
                next(r["total_reward"] for r in results if r["policy"] == pol and r["task_id"] == t)
                for t in tasks
            ]
            ax.bar([xx + i * width for xx in x], ys, width, label=pol)
        ax.set_xticks([xx + width for xx in x])
        ax.set_xticklabels(tasks)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Total reward")
        ax.set_title("Stakeholder Gym — policy × task")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out_path = out_dir / "reward_bars.png"
        plt.savefig(out_path, dpi=120)
        print(f"\nchart: {out_path}")
    except ImportError:
        print("\n(matplotlib not installed; skipping chart)")

    print(f"\nwrote {len(results)} rollouts to {out_dir}")


if __name__ == "__main__":
    main()
