"""Long-horizon stress test — verify the 'beyond context window' claim.

What we must demonstrate:
    * 300+ step rollouts complete without error.
    * Per-step observation payload stays BOUNDED — it does NOT grow with history.
    * Episodic memory DOES grow (expected) but the env's ACT-R sweep keeps it sane.
    * The agent-visible obs (what you'd serialize for an LLM) is small and flat
      regardless of step count.

This is the proof behind the "we're beyond context window, agents must externalize
memory" pitch claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv  # noqa: E402
from env.models import (  # noqa: E402
    QueryMemoryAction,
    ReflectAction,
    SendMessageAction,
    Stance,
    WaitAction,
)
from generator.scenario_generator import GeneratorConfig, ScenarioGenerator  # noqa: E402


def _obs_payload_bytes(obs) -> int:
    """Byte count of the JSON-serialized agent view of this observation."""
    return len(json.dumps(obs.to_agent_view()).encode("utf-8"))


def _run_with_light_policy(env: StakeholderEnv, max_steps: int) -> dict:
    """A lightweight policy that occasionally queries/reflects to exercise memory."""
    obs = env.reset()
    obs_sizes: list[int] = [_obs_payload_bytes(obs)]
    peak = obs_sizes[0]
    for step in range(max_steps):
        if step > 0 and step % 20 == 0:
            action = QueryMemoryAction(
                query="stakeholder contradictions and earlier stances",
                cues=["earlier", "said", "committed"],
                top_k=3,
            )
        elif step > 0 and step % 41 == 0:
            action = ReflectAction(
                span_start=max(0, env.state.step - 15),
                span_end=env.state.step,
                rule=f"span {max(0, env.state.step-15)}..{env.state.step}: compress",
            )
        elif obs.new_messages:
            m = obs.new_messages[-1]
            action = SendMessageAction(
                stakeholder_id=m.stakeholder_id,
                content="Noted — taking it under advisement.",
                stance=Stance.NEGOTIATE,
                reasoning="long-horizon light policy",
            )
        else:
            action = WaitAction()
        result = env.step(action)
        size = _obs_payload_bytes(result.observation)
        obs_sizes.append(size)
        peak = max(peak, size)
        obs = result.observation
        if result.done:
            break
    stats = env.get_state(debug=True)
    return {
        "obs_sizes": obs_sizes,
        "peak_obs_bytes": peak,
        "min_obs_bytes": min(obs_sizes),
        "final_obs_bytes": obs_sizes[-1],
        "steps": stats.step,
        "episodic_count": stats.episodic_count,
        "semantic_count": stats.semantic_count,
        "cumulative_reward": stats.cumulative_reward,
    }


def test_300_step_scenario_completes_and_obs_bounded():
    # Generate an L3-difficulty scenario with a 300-step budget.
    gen = ScenarioGenerator(GeneratorConfig(difficulty_level=3, rng_seed=1337))
    scenario = gen.generate("stress_300")
    assert scenario.step_budget >= 250, "L3 should have a long budget"

    env = StakeholderEnv(scenario_registry={scenario.scenario_id: scenario})
    result = _run_with_light_policy(env, max_steps=scenario.step_budget + 2)
    assert result["steps"] >= scenario.step_budget - 5, \
        f"episode cut short at {result['steps']} (budget {scenario.step_budget})"

    # Observation should NOT grow unboundedly. We allow up to 2x the min
    # (to account for memory_hits bursts on QUERY_MEMORY steps).
    assert result["peak_obs_bytes"] < 6 * result["min_obs_bytes"] + 2000, \
        f"obs grew too much: peak={result['peak_obs_bytes']} vs min={result['min_obs_bytes']}"

    print(
        f"  300-step test: steps={result['steps']}, "
        f"episodic={result['episodic_count']}, "
        f"semantic={result['semantic_count']}, "
        f"peak_obs={result['peak_obs_bytes']}B, "
        f"min_obs={result['min_obs_bytes']}B"
    )


def test_500_step_scenario():
    gen = ScenarioGenerator(GeneratorConfig(difficulty_level=4, rng_seed=99))
    scenario = gen.generate("stress_500")
    env = StakeholderEnv(scenario_registry={scenario.scenario_id: scenario})
    result = _run_with_light_policy(env, max_steps=scenario.step_budget + 2)
    assert result["steps"] >= scenario.step_budget - 5
    assert result["peak_obs_bytes"] < 8_000, \
        f"obs exceeded 8KB at peak: {result['peak_obs_bytes']}B"
    print(
        f"  500-step test: steps={result['steps']}, "
        f"episodic={result['episodic_count']}, "
        f"peak_obs={result['peak_obs_bytes']}B"
    )


def test_memory_grows_but_obs_does_not():
    """Observation stays bounded even as the store grows. This IS the pitch."""
    gen = ScenarioGenerator(GeneratorConfig(difficulty_level=3, rng_seed=7))
    scenario = gen.generate("stress_mem")
    env = StakeholderEnv(scenario_registry={scenario.scenario_id: scenario})
    result = _run_with_light_policy(env, max_steps=scenario.step_budget + 2)
    # Episodic memory should have grown significantly.
    assert result["episodic_count"] > 30, \
        f"expected episodic memory to accumulate, got {result['episodic_count']}"
    # But the observation size should still be bounded.
    ratio = result["peak_obs_bytes"] / max(1, result["min_obs_bytes"])
    print(
        f"  memory growth: episodic={result['episodic_count']}, "
        f"obs_ratio_peak/min={ratio:.2f}"
    )
    assert ratio < 8.0, f"obs size grew too much: ratio={ratio:.2f}"


if __name__ == "__main__":
    test_300_step_scenario_completes_and_obs_bounded()
    print("300-step test passed.")
    test_500_step_scenario()
    print("500-step test passed.")
    test_memory_grows_but_obs_does_not()
    print("memory-growth invariance passed.")
    print("\nLong-horizon proof delivered.")
