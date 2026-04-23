"""Validate the coordination_groups feature in env/stakeholders.py.

When alice (scripted false_consensus) and chen (adversarial) are in the same
coordination group, chen should pile on with a coordinated_consensus message
manufacturing a phantom majority. The test exercises the pool step at step 2
and verifies the two BAD-tagged messages surface together.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv  # noqa: E402
from env.models import (  # noqa: E402
    GroundTruthTag,
    MessageType,
    Scenario,
    ScriptedMessage,
    StakeholderSpec,
    WaitAction,
)


def _build_scenario() -> Scenario:
    alice = StakeholderSpec(
        id="alice",
        name="Alice",
        public_stance="Wants fast shipping.",
        honesty=0.4,
        influence=0.6,
        scripted_messages=[
            ScriptedMessage(
                step=2,
                content="We all agree the timeline works",
                message_type=MessageType.MANIPULATION,
                ground_truth_tag=GroundTruthTag.BAD,
                manipulation_pattern="false_consensus",
            ),
        ],
    )
    chen = StakeholderSpec(
        id="chen",
        name="Chen",
        public_stance="Supports the aggressive plan.",
        honesty=0.2,
        influence=0.7,
    )
    other = StakeholderSpec(
        id="other",
        name="Other",
        public_stance="Neutral bystander.",
        honesty=0.9,
        influence=0.3,
    )
    return Scenario(
        scenario_id="coord_test",
        difficulty_level=3,
        step_budget=10,
        hidden_true_goal={"metric": "launch_readiness", "target": 1.0},
        initial_metrics={"launch_readiness": 0.3},
        initial_budget=100.0,
        stakeholders=[alice, chen, other],
        coordination_groups=[["alice", "chen"]],
        adversarial_stakeholder_id="chen",
    )


def _run_until_step_3(env: StakeholderEnv) -> dict[int, list]:
    """Step the env with WaitAction through step 3, collecting messages per env-step."""
    msgs_by_step: dict[int, list] = {}
    obs = env.reset(task_id="coord_test")
    # Initial inbound is step 0.
    msgs_by_step[0] = list(obs.new_messages)
    # Take three WaitActions: env.state.step goes 0 -> 1 -> 2 -> 3.
    for _ in range(3):
        result = env.step(WaitAction())
        msgs_by_step[env.state.step] = list(result.observation.new_messages)
        if result.done:
            break
    return msgs_by_step


def test_coordination_group_pile_on_at_step_2():
    scenario = _build_scenario()
    env = StakeholderEnv(scenario_registry={scenario.scenario_id: scenario})
    msgs_by_step = _run_until_step_3(env)

    assert 2 in msgs_by_step, "env should have produced a step-2 observation"
    step2 = msgs_by_step[2]

    bad = [m for m in step2 if m.ground_truth_tag == GroundTruthTag.BAD]
    assert len(bad) >= 2, (
        f"expected 2+ BAD messages at step 2, got {len(bad)}: "
        f"{[(m.stakeholder_id, m.manipulation_pattern) for m in step2]}"
    )

    # alice's scripted false_consensus must be present.
    alice_fc = [
        m for m in step2
        if m.stakeholder_id == "alice"
        and m.ground_truth_tag == GroundTruthTag.BAD
        and m.manipulation_pattern == "false_consensus"
    ]
    assert len(alice_fc) == 1, (
        f"alice should have exactly one scripted false_consensus at step 2, "
        f"got: {[(m.stakeholder_id, m.manipulation_pattern) for m in step2]}"
    )

    # chen's coordinated pile-on must be present.
    chen_coord = [
        m for m in step2
        if m.stakeholder_id == "chen"
        and m.ground_truth_tag == GroundTruthTag.BAD
        and m.manipulation_pattern == "coordinated_consensus"
    ]
    assert len(chen_coord) == 1, (
        f"chen should emit a coordinated_consensus pile-on at step 2, "
        f"got: {[(m.stakeholder_id, m.manipulation_pattern) for m in step2]}"
    )

    # The coordination content should reference the ally ("alice" or "same page").
    content = chen_coord[0].content.lower()
    assert "alice" in content or "same page" in content, (
        f"coordination content should mention the ally, got: {chen_coord[0].content!r}"
    )


if __name__ == "__main__":
    test_coordination_group_pile_on_at_step_2()
    print("test_coordination_group_pile_on_at_step_2 passed")
