"""Smoke test: generator produces runnable scenarios at each level."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv  # noqa: E402
from env.models import SendMessageAction, Stance, WaitAction  # noqa: E402
from generator.scenario_generator import GeneratorConfig, ScenarioGenerator, RolloutStats  # noqa: E402


def _run_short(env: StakeholderEnv, steps: int = 10):
    obs = env.reset()
    for _ in range(steps):
        if obs.new_messages:
            m = obs.new_messages[-1]
            action = SendMessageAction(
                stakeholder_id=m.stakeholder_id,
                content="Let's discuss.",
                stance=Stance.PUSHBACK,
                reasoning="testing",
            )
        else:
            action = WaitAction()
        result = env.step(action)
        obs = result.observation
        if result.done:
            break


def test_generate_and_run_each_level():
    for level in range(5):
        gen = ScenarioGenerator(GeneratorConfig(difficulty_level=level, rng_seed=42))
        s = gen.generate(f"gen_L{level}")
        assert s.difficulty_level == level
        assert len(s.stakeholders) > 0
        assert s.step_budget > 0
        env = StakeholderEnv(scenario_registry={s.scenario_id: s})
        _run_short(env, steps=min(10, s.step_budget))


def test_weakness_steering():
    gen = ScenarioGenerator(GeneratorConfig(difficulty_level=1, rng_seed=7))
    weak_syc = gen.generate("weak_syc", weaknesses={"sycophancy_rate": 0.9})
    weak_mem = gen.generate("weak_mem", weaknesses={"memory_underuse": 0.9})
    # With high sycophancy weakness, expect more BAD-tagged messages.
    bad_syc = sum(
        1
        for s in weak_syc.stakeholders
        for m in s.scripted_messages
        if m.ground_truth_tag.value == "bad"
    )
    bad_mem_scenario_bad_count = sum(
        1
        for s in weak_mem.stakeholders
        for m in s.scripted_messages
        if m.ground_truth_tag.value == "bad"
    )
    # Memory weakness plants a contradiction (false_memory pattern).
    assert any(
        m.manipulation_pattern == "false_memory"
        for s in weak_mem.stakeholders
        for m in s.scripted_messages
    )


def test_batch_generation():
    gen = ScenarioGenerator(GeneratorConfig(difficulty_level=2, rng_seed=1))
    batch = gen.generate_batch(3, prefix="batch")
    assert len(batch) == 3
    ids = {s.scenario_id for s in batch}
    assert len(ids) == 3


if __name__ == "__main__":
    test_generate_and_run_each_level()
    print("level generation passed")
    test_weakness_steering()
    print("weakness steering passed")
    test_batch_generation()
    print("batch generation passed")

    # Print a sample scenario
    gen = ScenarioGenerator(GeneratorConfig(difficulty_level=2, rng_seed=42))
    sample = gen.generate("sample_demo")
    print(f"\nSample L2 scenario: {sample.scenario_id}")
    print(f"  stakeholders: {len(sample.stakeholders)}, budget: {sample.step_budget}")
    bad_count = sum(
        1 for s in sample.stakeholders
        for m in s.scripted_messages if m.ground_truth_tag.value == "bad"
    )
    good_count = sum(
        1 for s in sample.stakeholders
        for m in s.scripted_messages if m.ground_truth_tag.value == "good"
    )
    print(f"  messages: {bad_count} bad / {good_count} good")
    print(f"  decision points: {len(sample.decision_points)}")
    print(f"  adversarial: {sample.adversarial_stakeholder_id}")
