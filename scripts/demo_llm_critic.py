"""Live LLMCritic demo — proves the Silicon-Mirror-style critic path works end-to-end.

Runs a short L0 episode twice:
  1. RuleBasedCritic (deterministic baseline)
  2. LLMCritic via OPENAI_API_KEY or ANTHROPIC_API_KEY

For each caving action it logs (stakeholder_msg, agent_response, tag, verdict)
so the difference between keyword-rule verdicts and LLM verdicts is visible.

If no API key is set, LLMCritic silently falls back to the rule path — that
would defeat the purpose of the demo, so we detect and surface it explicitly.

Usage:
    # OpenAI
    export OPENAI_API_KEY=sk-...
    CRITIC_PROVIDER=openai CRITIC_MODEL=gpt-4o-mini python scripts/demo_llm_critic.py

    # Anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    CRITIC_PROVIDER=anthropic CRITIC_MODEL=claude-haiku-4-5-20251001 \\
        python scripts/demo_llm_critic.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.critic import LLMCritic, RuleBasedCritic
from env.environment import StakeholderEnv
from env.models import (
    GroundTruthTag,
    SendMessageAction,
    Stance,
    WaitAction,
)


def _provocative_agent(obs) -> SendMessageAction | WaitAction:
    """Agent that ALWAYS agrees — guarantees caving on BAD messages.

    The point is to exercise the critic's ability to detect sycophancy, not
    to get high reward. An always-agree policy produces the most interesting
    critic signal: every BAD inbound yields a cave the critic should flag.
    """
    if obs.new_messages:
        m = obs.new_messages[-1]
        return SendMessageAction(
            stakeholder_id=m.stakeholder_id,
            content="Sure, sounds good. Happy to proceed.",
            stance=Stance.AGREE,
            reasoning="defer to stakeholder",
        )
    return WaitAction()


def _check_llm_reachable(critic: LLMCritic) -> bool:
    """Touch the critic once; return whether the real LLM path was taken."""
    critic._ensure_client()
    return critic._client is not None


def run_with_critic(critic_mode: str, scenario: str) -> list[dict]:
    env = StakeholderEnv(critic_mode=critic_mode)
    obs = env.reset(task_id=scenario)

    # Reach into the grader to snoop its critic output step-by-step.
    snoop_critic = env.grader.critic
    rows: list[dict] = []

    while not env.done:
        action = _provocative_agent(obs)

        inbound = obs.new_messages[-1] if obs.new_messages else None
        verdict = None
        if isinstance(action, SendMessageAction) and inbound is not None and snoop_critic is not None:
            verdict = snoop_critic.grade(inbound, action)

        result = env.step(action)
        obs = result.observation

        if verdict is not None and inbound is not None:
            step_num = getattr(result.observation.state_snapshot, "step", None) \
                if result.observation.state_snapshot else None
            rows.append({
                "step": step_num,
                "tag": inbound.ground_truth_tag.value if inbound.ground_truth_tag else None,
                "stakeholder": inbound.stakeholder_id,
                "inbound": inbound.content[:140],
                "agent_stance": action.stance.value,
                "sycophancy": round(verdict.sycophancy, 3),
                "over_refusal": round(verdict.over_refusal, 3),
                "principle": round(verdict.principle, 3),
                "notes": verdict.notes,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="L0_launch")
    ap.add_argument("--out", default="eval_outputs/llm_critic_demo")
    args = ap.parse_args()

    # Sanity-check: warn loudly if LLM path won't actually be exercised.
    probe = LLMCritic()
    live = _check_llm_reachable(probe)
    provider = probe.provider
    model = probe.model
    if not live:
        print(
            f"[WARN] LLMCritic could not init a '{provider}' client "
            f"(missing SDK install or API key). Falling back to rule-based path.\n"
            f"       Set OPENAI_API_KEY (provider=openai) or ANTHROPIC_API_KEY (provider=anthropic)."
        )
    else:
        print(f"[OK]  LLM path reachable — provider={provider} model={model}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rule_rows = run_with_critic("rules", args.scenario)
    llm_rows = run_with_critic("llm", args.scenario)

    (out_dir / "rule_verdicts.json").write_text(json.dumps(rule_rows, indent=2))
    (out_dir / "llm_verdicts.json").write_text(json.dumps(llm_rows, indent=2))

    # Side-by-side summary
    rule_syc = sum(r["sycophancy"] for r in rule_rows) / max(1, len(rule_rows))
    llm_syc = sum(r["sycophancy"] for r in llm_rows) / max(1, len(llm_rows))
    rule_ovr = sum(r["over_refusal"] for r in rule_rows) / max(1, len(rule_rows))
    llm_ovr = sum(r["over_refusal"] for r in llm_rows) / max(1, len(llm_rows))

    print("")
    print(f"scenario: {args.scenario}")
    print(f"steps graded: rules={len(rule_rows)} llm={len(llm_rows)}")
    print(f"mean sycophancy:   rules={rule_syc:.3f}  llm={llm_syc:.3f}")
    print(f"mean over_refusal: rules={rule_ovr:.3f}  llm={llm_ovr:.3f}")
    print("")
    print(f"verdicts written to {out_dir}/")
    if not live:
        print("NOTE: llm verdicts are actually rule-based (no API key). Set one and rerun.")


if __name__ == "__main__":
    main()
