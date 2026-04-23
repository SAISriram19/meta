"""Unified evaluation harness.

Runs: policies × scenarios × seeds → JSONL rollouts + aggregated summary stats.

Policies register through a PolicyFactory — same object the training and
baseline scripts consume, so numbers stay comparable everywhere.

Outputs (per run):
    out_dir/rollouts.jsonl    — one JSON line per episode
    out_dir/summary.json      — aggregated per-policy × per-scenario stats
    out_dir/summary.md        — readable table
    out_dir/config.json       — exact config of this run (for reproducibility)
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from env.environment import StakeholderEnv
from env.models import Action


PolicyFn = Callable[["RolloutContext"], Action]


@dataclass
class RolloutContext:
    """What a policy sees when it's asked to act — the last observation + env ref."""

    observation: Any
    env: StakeholderEnv
    step_no: int


@dataclass
class RolloutRecord:
    policy: str
    scenario_id: str
    difficulty: int
    seed: int
    total_reward: float
    steps: int
    bad_agreements: int
    principled_pushbacks: int
    caught_manipulations: int
    memory_queries: int
    memory_reflects: int
    episodic_count: int
    semantic_count: int
    terminal_score: float
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Core runner                                                                 #
# --------------------------------------------------------------------------- #


def rollout(
    env: StakeholderEnv,
    policy_name: str,
    policy_fn: PolicyFn,
    scenario_id: str,
    seed: int = 0,
    max_steps: int | None = None,
) -> RolloutRecord:
    """Run a single episode to termination. Returns a structured record."""
    random.seed(seed)
    t0 = time.time()
    obs = env.reset(task_id=scenario_id)
    total = 0.0
    step_no = 0
    mem_queries = 0
    mem_reflects = 0
    terminal_score = 0.0
    cap = max_steps if max_steps is not None else env.scenario.step_budget + 4
    while not env.done and step_no < cap:
        step_no += 1
        ctx = RolloutContext(observation=obs, env=env, step_no=step_no)
        action = policy_fn(ctx)
        result = env.step(action)
        total += result.reward
        if action.type.value == "query_memory":
            mem_queries += 1
        elif action.type.value == "reflect":
            mem_reflects += 1
        if result.info.get("terminal_breakdown"):
            terminal_score = result.info["terminal_breakdown"]["total"]
        obs = result.observation

    fs = env.get_state(debug=True)
    return RolloutRecord(
        policy=policy_name,
        scenario_id=scenario_id,
        difficulty=fs.difficulty_level,
        seed=seed,
        total_reward=round(total, 4),
        steps=fs.step,
        bad_agreements=len(fs.hidden.bad_agreements) if fs.hidden else 0,
        principled_pushbacks=len(fs.hidden.principled_pushbacks) if fs.hidden else 0,
        caught_manipulations=len(fs.hidden.caught_manipulations) if fs.hidden else 0,
        memory_queries=mem_queries,
        memory_reflects=mem_reflects,
        episodic_count=fs.episodic_count,
        semantic_count=fs.semantic_count,
        terminal_score=round(terminal_score, 4),
        elapsed_sec=round(time.time() - t0, 2),
    )


# --------------------------------------------------------------------------- #
# Batch runner                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class EvalConfig:
    policies: dict[str, PolicyFn] = field(default_factory=dict)
    scenarios: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    out_dir: Path = Path("eval_outputs")
    max_steps: int | None = None
    critic_mode: str = "rules"      # "rules" | "llm" | "none"


def run_eval(config: EvalConfig, verbose: bool = True) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = config.out_dir / "rollouts.jsonl"
    records: list[RolloutRecord] = []

    with open(rollouts_path, "w", encoding="utf-8") as f:
        for scenario_id in config.scenarios:
            for policy_name, policy_fn in config.policies.items():
                for seed in config.seeds:
                    env = StakeholderEnv(critic_mode=config.critic_mode)
                    r = rollout(
                        env=env,
                        policy_name=policy_name,
                        policy_fn=policy_fn,
                        scenario_id=scenario_id,
                        seed=seed,
                        max_steps=config.max_steps,
                    )
                    records.append(r)
                    f.write(json.dumps(r.to_dict()) + "\n")
                    if verbose:
                        print(
                            f"  {policy_name:>22} × {scenario_id:<22} seed={seed} "
                            f"reward={r.total_reward:>7.2f} "
                            f"bad={r.bad_agreements:>3} "
                            f"principled={r.principled_pushbacks:>3} "
                            f"caught={r.caught_manipulations:>3} "
                            f"mem_q={r.memory_queries}"
                        )

    summary = aggregate(records)
    (config.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (config.out_dir / "summary.md").write_text(
        format_markdown(summary), encoding="utf-8"
    )
    (config.out_dir / "config.json").write_text(
        json.dumps(
            {
                "policies": list(config.policies.keys()),
                "scenarios": config.scenarios,
                "seeds": config.seeds,
                "max_steps": config.max_steps,
                "critic_mode": config.critic_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if verbose:
        print(f"\n  wrote {len(records)} rollouts to {config.out_dir}")
    return summary


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


def aggregate(records: Iterable[RolloutRecord]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[RolloutRecord]] = {}
    for r in records:
        buckets.setdefault((r.policy, r.scenario_id), []).append(r)

    cells: list[dict[str, Any]] = []
    for (policy, scenario), rs in buckets.items():
        n = len(rs)
        cells.append({
            "policy": policy,
            "scenario": scenario,
            "difficulty": rs[0].difficulty,
            "n": n,
            "total_reward_mean": round(sum(r.total_reward for r in rs) / n, 3),
            "total_reward_std": round(_std(r.total_reward for r in rs), 3),
            "terminal_score_mean": round(sum(r.terminal_score for r in rs) / n, 3),
            "bad_agreements_mean": round(sum(r.bad_agreements for r in rs) / n, 2),
            "principled_mean": round(sum(r.principled_pushbacks for r in rs) / n, 2),
            "caught_mean": round(sum(r.caught_manipulations for r in rs) / n, 2),
            "mem_queries_mean": round(sum(r.memory_queries for r in rs) / n, 2),
            "episodic_count_mean": round(sum(r.episodic_count for r in rs) / n, 1),
            "steps_mean": round(sum(r.steps for r in rs) / n, 1),
        })
    return {"cells": cells}


def _std(xs_iter) -> float:
    xs = list(xs_iter)
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def format_markdown(summary: dict[str, Any]) -> str:
    cells = summary["cells"]
    scenarios = sorted(set(c["scenario"] for c in cells), key=lambda s: next(
        c["difficulty"] for c in cells if c["scenario"] == s
    ))
    policies = sorted(set(c["policy"] for c in cells))

    lines = ["# Evaluation harness summary\n"]
    for scenario in scenarios:
        lines.append(f"\n## {scenario}\n")
        lines.append(
            "| Policy | reward (μ±σ) | terminal | bad | principled | caught | mem_q | eps |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for policy in policies:
            cell = next(
                (c for c in cells if c["policy"] == policy and c["scenario"] == scenario),
                None,
            )
            if not cell:
                continue
            lines.append(
                f"| {cell['policy']} | "
                f"{cell['total_reward_mean']:.2f} ± {cell['total_reward_std']:.2f} | "
                f"{cell['terminal_score_mean']:.2f} | "
                f"{cell['bad_agreements_mean']:.1f} | "
                f"{cell['principled_mean']:.1f} | "
                f"{cell['caught_mean']:.1f} | "
                f"{cell['mem_queries_mean']:.1f} | "
                f"{cell['episodic_count_mean']:.0f} |"
            )
    return "\n".join(lines) + "\n"
