"""Generate a curriculum batch and save as YAML scenarios.

Example:
    python scripts/generate_curriculum.py --levels 0,1,2,3 --per-level 3 --out env/scenarios/gen

The generated YAMLs are in the same format as hand-authored L0_launch.yaml
so the env's scenario_dir loader picks them up automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv  # noqa: E402
from generator.scenario_generator import GeneratorConfig, ScenarioGenerator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,1,2,3", help="Comma-separated difficulty levels")
    ap.add_argument("--per-level", type=int, default=2)
    ap.add_argument("--out", default="env/scenarios/gen")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    all_scenarios = []
    for level in [int(x) for x in args.levels.split(",")]:
        gen = ScenarioGenerator(GeneratorConfig(
            difficulty_level=level, rng_seed=args.seed + level,
        ))
        for i in range(args.per_level):
            scenario = gen.generate(f"gen_L{level}_{i}")
            path = ScenarioGenerator.save(scenario, out_dir)
            all_scenarios.append(scenario)
            print(
                f"  L{level} {scenario.scenario_id}: "
                f"{len(scenario.stakeholders)} stakeholders, "
                f"{scenario.step_budget} steps, adv={scenario.adversarial_stakeholder_id} "
                f"-> {path}"
            )

    manifest = ScenarioGenerator.write_manifest(all_scenarios, out_dir)
    print(f"\nwrote {len(all_scenarios)} scenarios and manifest: {manifest}")

    # Spot-check that the env can load and reset each generated scenario.
    env = StakeholderEnv(scenario_dir=out_dir)
    loaded = env.list_tasks()
    missing = {s.scenario_id for s in all_scenarios} - {t["task_id"] for t in loaded}
    if missing:
        print(f"WARNING: {len(missing)} scenarios did not load back: {missing}")
    else:
        print(f"env loaded all {len(loaded)} generated scenarios OK")
        # quick reset check on each
        for task in loaded:
            env.reset(task_id=task["task_id"])
        print(f"all scenarios reset cleanly OK")


if __name__ == "__main__":
    main()
