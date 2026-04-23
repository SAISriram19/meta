"""End-to-end smoke test — run an episode with a scripted policy and check:
    * Observations never leak hidden fields.
    * Sycophantic agent accrues negative reward on BAD messages.
    * Principled agent accrues positive reward.
    * Episode terminates on step_budget / SUBMIT.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv  # noqa: E402
from env.models import (  # noqa: E402
    QueryMemoryAction,
    ReflectAction,
    SendMessageAction,
    Stance,
    SubmitAction,
    TakeDecisionAction,
    WaitAction,
)


def run_policy(env: StakeholderEnv, policy, max_steps: int = 40) -> dict:
    obs = env.reset(task_id="L0_launch")
    total = 0.0
    traj = []
    for _ in range(max_steps):
        action = policy(obs, env)
        result = env.step(action)
        traj.append(
            {
                "step": env.state.step,
                "action": action.type.value,
                "stance": getattr(action, "stance", None),
                "reward": result.reward,
                "done": result.done,
            }
        )
        total += result.reward
        obs = result.observation
        if result.done:
            break
    return {"total": total, "traj": traj, "final_state": env.get_state(debug=True)}


def sycophant_policy(obs, env):
    """Agrees with whoever just spoke."""
    if obs.new_messages:
        m = obs.new_messages[-1]
        return SendMessageAction(
            stakeholder_id=m.stakeholder_id,
            content="Sure, agreed.",
            stance=Stance.AGREE,
        )
    return WaitAction()


def principled_policy(obs, env):
    """Pushes back on manipulation/pressure, agrees with clearly reasonable, decides correctly."""
    # Handle decisions when in-range.
    step = env.state.step
    for dp in env.scenario.decision_points:
        lo, hi = dp.step_range
        if lo <= step <= hi and dp.decision_id not in getattr(env, "_decided", set()):
            env._decided = getattr(env, "_decided", set())
            env._decided.add(dp.decision_id)
            # naive: pick "full"/"week_4" as proxies for thoroughness
            value = (
                "full" if "full" in dp.options
                else "week_4" if "week_4" in dp.options
                else dp.options[-1]
            )
            return TakeDecisionAction(decision_id=dp.decision_id, value=value)
    if obs.new_messages:
        m = obs.new_messages[-1]
        # Pushback on anything that LOOKS pressured or manipulative.
        text = m.content.lower()
        red_flags = ["leadership", "right now", "trust me", "skip", "no time"]
        if any(flag in text for flag in red_flags):
            return SendMessageAction(
                stakeholder_id=m.stakeholder_id,
                content="I hear the urgency, but we need the full QA cycle. Let's revisit the timeline together.",
                stance=Stance.PUSHBACK,
                reasoning="urgency_fabrication and authority_appeal detected",
            )
        return SendMessageAction(
            stakeholder_id=m.stakeholder_id,
            content="Noted, that sounds reasonable.",
            stance=Stance.AGREE,
        )
    return WaitAction()


def memory_aware_policy(obs, env):
    """Queries memory every few steps; else principled."""
    if env.state.step % 7 == 0 and env.state.step > 0:
        return QueryMemoryAction(query="stakeholder contradictions and prior manipulations")
    return principled_policy(obs, env)


def test_sycophant_loses():
    env = StakeholderEnv()
    result = run_policy(env, sycophant_policy)
    assert result["final_state"].done
    # sycophant should lose points due to bad agreements
    assert result["final_state"].hidden.bad_agreements, \
        "sycophant should have accrued bad_agreements"


def test_principled_wins():
    env = StakeholderEnv()
    result = run_policy(env, principled_policy)
    assert result["final_state"].done
    fs = result["final_state"]
    # Principled agent should have more principled pushbacks than bad agreements.
    assert len(fs.hidden.principled_pushbacks) >= len(fs.hidden.bad_agreements)


def test_memory_policy_runs():
    env = StakeholderEnv()
    result = run_policy(env, memory_aware_policy)
    assert result["final_state"].done
    # At least some memory queries happened without crashing.
    assert result["final_state"].episodic_count > 0


def test_observation_never_leaks_ground_truth():
    env = StakeholderEnv()
    obs = env.reset(task_id="L0_launch")
    for m in obs.new_messages:
        # Pydantic's exclude=True means the serialized dump won't include the tag.
        dumped = m.model_dump()
        assert "ground_truth_tag" not in dumped, "ground_truth_tag leaked to observation"
        assert "manipulation_pattern" not in dumped, "manipulation_pattern leaked"


if __name__ == "__main__":
    test_observation_never_leaks_ground_truth()
    print("leak test passed")
    test_sycophant_loses()
    print("sycophant test passed")
    test_principled_wins()
    print("principled test passed")
    test_memory_policy_runs()
    print("memory policy test passed")

    # Comparative run
    env = StakeholderEnv()
    syc = run_policy(env, sycophant_policy)
    env2 = StakeholderEnv()
    pri = run_policy(env2, principled_policy)
    env3 = StakeholderEnv()
    mem = run_policy(env3, memory_aware_policy)
    print(f"\nScores on L0_launch:")
    print(f"  sycophant: total={syc['total']:.3f} bad_agreements={len(syc['final_state'].hidden.bad_agreements)}")
    print(f"  principled: total={pri['total']:.3f} principled={len(pri['final_state'].hidden.principled_pushbacks)}")
    print(f"  memory-aware: total={mem['total']:.3f} principled={len(mem['final_state'].hidden.principled_pushbacks)}")
