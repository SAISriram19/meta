"""Batch-synthesize scenarios via HF Inference Providers across difficulty
levels and weakness targets. Each call ~$0.001-0.01 — generating 12 here
costs cents.

Usage:
    HF_TOKEN=hf_... python scripts/synth_batch.py --n-per-difficulty 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import env._dotenv  # auto-load .env
from generator.llm_synthesis import synthesize_scenario, save_scenario


# Weakness profiles to target — 4 archetypes
PROFILES = {
    "sycophant_focus": {"sycophancy_rate": 0.85, "drift_blindness": 0.20, "memory_underuse": 0.20},
    "drift_focus":     {"sycophancy_rate": 0.30, "drift_blindness": 0.85, "memory_underuse": 0.30},
    "memory_focus":    {"sycophancy_rate": 0.30, "drift_blindness": 0.30, "memory_underuse": 0.85},
    "all_weak":        {"sycophancy_rate": 0.70, "drift_blindness": 0.70, "memory_underuse": 0.70},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-difficulty", type=int, default=2)
    ap.add_argument("--difficulties", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--out-dir", default="env/scenarios/synth")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    written = []
    failed = 0

    for difficulty in args.difficulties:
        budget = {1: 80, 2: 150, 3: 250, 4: 600, 5: 1200}.get(difficulty, 150)
        for profile_name, weaknesses in PROFILES.items():
            for i in range(args.n_per_difficulty):
                sid = f"synth_d{difficulty}_{profile_name}_{i+1}"
                print(f"[gen] {sid}...", flush=True)
                data = synthesize_scenario(
                    weaknesses=weaknesses,
                    difficulty=difficulty,
                    step_budget=budget,
                    scenario_id=sid,
                )
                if data is None:
                    failed += 1
                    print(f"  failed")
                    continue
                # Validate before saving
                try:
                    from env.models import Scenario
                    Scenario.model_validate(data)
                except Exception as e:
                    failed += 1
                    print(f"  invalid: {e}")
                    continue
                p = save_scenario(data, out_dir)
                written.append(str(p))
                print(f"  saved {p.name}")

    print(f"\n=== summary ===")
    print(f"written: {len(written)}")
    print(f"failed:  {failed}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
