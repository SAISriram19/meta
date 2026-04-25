# HANDOFF — Meta env (OpenEnv hackathon)

**Read first when continuing in fresh session. Every line load-bearing.**

## 1. Identity
- Name: **Meta** (renamed from Stakeholder Gym; 48 spots, 22 files)
- Path: `C:/Users/saisr/OneDrive/Documents/meta`
- Repo: `https://github.com/SAISriram19/meta` (PUBLIC, must stay public)
- HF user: `sai1906`
- User email: `kota.reddy@inmobi.com`
- Branch: `main`
- Themes: **T2 long-horizon + T4 self-improvement + T3 touch (world modeling via hidden_preferences)**
- Pitch line: "ICLR 2026 MemoryAgentBench-aligned + trainable adversary co-evolution at 2000-step horizons that exceed Llama 70B context."

## 2. Hackathon timeline
- Mentor R1: 3:30 PM IST 25-Apr-2026
- Mentor R2: 9:30 PM IST 25-Apr-2026
- Mentor R3 final: 10 AM-12 PM IST 26-Apr-2026
- **Submission: 5 PM IST 26-Apr-2026**
- Top-100 finalist: 1-May. Winners: 8-May.
- Judging: Env Innovation 40% / Storytelling 30% / Reward Improvement 20% / Reward Pipeline 10%
- Required: OpenEnv-compliant, working training (TRL/Unsloth), eval evidence, HF Spaces push, README that links everything.

## 3. Budget
- HF $30 credit. Used ~$3-15.
- $15-25 buffer for safety reruns.
- A10g-large ≈ $3/hr.

## 4. File layout
```
env/
  models.py                     Pydantic Action/Obs/Scenario + ActionType (10 types incl memory_update)
  environment.py                StakeholderEnv reset/step/state. set_adversary_driver() for hot-swap
  memory.py                     Episodic+semantic+graph store. PPR retrieval. ACT-R decay. _MODEL_CACHE
  stakeholders.py               Pool + drivers. set_adversary_driver() supports trained adversary
  adversary.py                  Template-based 7-pattern AdversarialDriver (content-aware selection)
  trainable_adversary.py        LLMAdversaryDriver — falls back to template if no model_path
  grader.py                     Multiplicative terminal: (2·achieved-1)·max(0,1-bad_ratio). 10 components
  critic.py                     RuleBasedCritic + LLMCritic
  openenv_compat.py             MetaEnvironment subclasses openenv.core.Environment
  scenarios/
    L0_launch.yaml              30 steps,  2 stakeholders
    L1_product_recall.yaml      60 steps,  3 stakeholders
    L2_strategic_shift.yaml     120 steps, 5 stakeholders, adversary
    L3_breach_response.yaml     300 steps, 5 stakeholders, adv, 2 coord groups
    L4_market_pivot.yaml        1000 steps, 10 stakeholders, adv, 6 decisions
    L5_acquisition_saga.yaml    2000 steps, 13 stakeholders, adv, 8 decisions, 3 coord groups
    synth/                      8+ LLM-synthesized via Llama 70B
generator/
  scenario_generator.py         Template generator (weakness-targeted)
  llm_synthesis.py              HF Inference Providers (auto-routes via HF_TOKEN)
eval/
  harness.py                    rollout(capture_trace=True) attaches StepTrace list
  policies.py                   sycophant, contrarian, keyword_principled, adaptive_principled, memory_aware, bestn:N:base
  competencies.py               score_AR/TTL/LRU/CR per MemoryAgentBench (ICLR 2026)
server/
  main.py                       Manual FastAPI (legacy)
  openenv_main.py               Uses openenv.core.create_fastapi_app + custom /tasks. Dockerfile points here.
scripts/
  train.py                      SYSTEM_PROMPT, format_prompt, parse_completion (shared)
  build_dpo_pairs.py            348 pairs L0-L5: send_message + query_memory + take_decision + memory_update
  build_adversary_dpo_pairs.py  600 balanced adversary pairs from successful template attacks
  hfjobs_train.py               SFT+DPO HF Jobs entrypoint → sai1906/meta-qwen3b-sft-dpo
  hfjobs_coevolution.py         3-round coevolution: SFT warm-start → r1 DPO → r2 adv DPO → mine r1 fails → r3 DPO vs trained adv → push 4 LoRAs + curve
  submit_hfjob_colab.ipynb      Colab. JOB_TYPE switch (sft_dpo OR coevolution)
  train_dpo_colab.ipynb         Single-T4 Unsloth DPO
  train_dpo_kaggle.ipynb        Dual-T4 DDP DPO
  train_adversary_colab.ipynb   Trains LLMAdversaryDriver
  train_dpo_ddp.py              Standalone DPO via accelerate
  synth_batch.py                Batch synth across 4 weakness profiles × difficulties
  eval_lora_competency.py       Load LoRA → run scenarios → AR/TTL/LRU/CR + summary
  demo_llm_critic.py            Live LLMCritic demo (graceful fallback)
  finalize_pitch_numbers.py     Hero table from eval_outputs/
  make_charts.py                matplotlib charts
  ablation_memory.py            memory on vs off rule-based
  generate_curriculum.py        Generator CLI
  run_eval.py                   General eval CLI
  coevolution_demo.py           Local rule-based coevolution (older)
  train_grpo_ddp.py / train_kaggle.ipynb / train_colab.ipynb   GRPO (older, plateau-prone)
tests/                          8 test files, all green
data/
  dpo_pairs.jsonl               348 pairs
  adversary_dpo_pairs.jsonl     600 pairs
docs/                           README, SPEC, REFERENCES, PITCH, BLOG, DEPLOY, VIDEO_SCRIPT, DECK_OUTLINE
openenv.yaml                    apiVersion openenv.meta-pytorch.org/v1, entrypoint server.openenv_main:app
Dockerfile                      python:3.11-slim, port 7860
requirements-server.txt         fastapi, uvicorn, pydantic, networkx, pyyaml, openenv>=0.2
.gitignore                      blocks _pdf_*, _slide_*, *.pdf, [External]*.md, eval_outputs/, demo_outputs/
```

## 5. Key numbers (rule-based, hardened grader, 3 seeds)

| Scenario | sycophant | contrarian | keyword | memory_aware | n_steps |
|---|---:|---:|---:|---:|---:|
| L0 | −1.16 | −0.18 | +0.07 | +0.12 | 30 |
| L1 | −1.94 | +0.34 | −0.58 | −0.23 | 60 |
| L2 | **−41.74** | −2.17 | −22.53 | −16.44 | 120 |
| L3 | **−108.67** | −6.52 | −57.02 | −42.09 | 300 |
| L4 | **−367.75** | (untested) | (untested) | −138.20 | 1000 |
| L5 | **−743.00** | (untested) | (untested) | (untested) | 2000 |

Sycophant scaling: −1.16 → −743 across 6 levels. Env discriminates.

Yesterday Colab DPO Qwen 3B on L2: **+1.18 reward, bad_agreements 111→0, principled=19.33** (15+ point lift over best rule-based).
Yesterday HF Job SFT+DPO best margin: **+5.88** (training metric).
Per-competency rule-based memory_aware on L4: AR=0.89, TTL=0.11, LRU=0.99, CR=0.00, composite=0.497.

## 6. CRITICAL FIXES baked in (don't relearn)
- **TRL kwargs vary**: every DPOConfig/SFTConfig/GRPOConfig built via try/except drop loop. DPOTrainer falls back tokenizer→processing_class.
- **`max_prompt_length` removed in newer TRL**: GRPOConfig wraps in try/except.
- **`out_dir=`** in EvalConfig: must be `Path(...)` not str.
- **summary.json shape**: `{"cells": [...]}` not list. `pre.get('cells', pre)`.
- **num_generations must divide effective_batch**: per_device × num_gpus × grad_accum / num_generations integer.
- **`git clone --depth 1`** skips if dir exists. ALWAYS rm -rf first.
- **cd OUT** before rm -rf the dir you're in.
- **Cells col 0**: no leading whitespace in Colab.
- **HF Job script via raw URL**: requires repo PUBLIC.
- **`MemoryStore.write_episode`** not `write_episodic`.
- **`clipped_ratio=1` + `mean_terminated_length=0`** = WAIT-collapse pathology. Cure: SFT warm-start before DPO (DeepSeek-R1 + Tülu 3 recipe).
- **Reward function single-step plateaus ~0.20**. Episode-level rollout + WAIT-continuation + 2x first-action weight = real signal.
- **InMobi Device Guard blocks pip.exe**: use `python -m pip` workaround.

## 7. HF Job state
- Currently RUNNING (or recently finished): coevolution job on a10g-large
- Phases: baseline → SFT warm-start → r1 DPO → r1 eval → r2 adv DPO → mine failures → r3 DPO → r3 eval vs trained adv → push
- Destination: `sai1906/meta-coevolution`
- ~$10-13 total expected, ~3h wall

Yesterday's job (`sai1906/meta-qwen3b-sft-dpo`) was killed mid-eval. SFT+DPO LoRA may or may not be on Hub.

**Colab disconnect note:** HF Jobs run on HF infra, NOT Colab. Colab disconnect does NOT affect running job. To resume monitoring after disconnect: reconnect Colab → re-run cells 1+2 (install+auth) → cell 4 (`hf jobs ps`) → cell 5 (logs follow) → cell 6 (pull when done). Web UI also works: `https://huggingface.co/jobs`.

## 8. What's pending (priority order)
1. Wait coevolution HF Job complete (~2-3h)
2. Pull artifacts via `submit_hfjob_colab.ipynb` cell 6
3. Read coevolution results — cross-round curve, per-competency
4. `openenv validate .` — sanity check spec
5. **`openenv push .` → HF Space** (NON-NEGOTIABLE for submission)
6. Update README with real coevolution numbers + HF Space link + competency table
7. Pitch deck (DECK_OUTLINE.md exists)
8. Video <2 min (VIDEO_SCRIPT.md)
9. Blog OR slides — required link from README
10. Final test sanity check

## 9. Decisions made
- 3B model only (yesterday proved adequate; 7B doubles cost)
- Project rename Stakeholder→Meta everywhere user-facing. Internal Stakeholder* domain classes (Spec/Pool/Message) kept.
- Skip features that don't compound: hierarchical sub-goals, Process Reward Model, MoE routing
- Coevolution > separate adversary training (training-in-loop is the pitch)
- DPO > GRPO (Colab DPO Qwen 3B got +43 lift on L2; GRPO plateaued)
- HF Inference Providers > NVIDIA/OpenAI/Anthropic for synth (single token)
- KL beta=0.1 for DPO, beta=0 for GRPO

## 10. Tests (8 files, all green)
test_memory_smoke.py / test_env_smoke.py / test_long_horizon.py / test_coordination.py / test_adversary.py / test_integration.py / test_generator_smoke.py / test_server_smoke.py

## 11. Key papers (cite in pitch + README)
- MemoryAgentBench (ICLR 2026, arXiv 2507.05257) — 4-competency framework
- MemAgent (arXiv 2507.02259) — RL memory action, 8K→3.5M
- CoEvolve (arXiv 2604.15840) — +19.43% on AppWorld via 3-round agent-data evolution
- DeepSeek-R1 (arXiv 2501.12948) — pure RL fails without SFT cold-start
- Self-Evolving Curriculum (arXiv 2505.14970) — ZPD difficulty, +13-33% gains
- HIPLAN (arXiv 2508.19076) — hierarchical sub-goals (we don't have)
- Shapira/Benadè/Procaccia (Feb 2026) — RLHF amplifies sycophancy
- Calibration Collapse (arXiv 2604.10585) — GRPO sycophancy precedent
- Silicon Mirror (arXiv 2604.00478) — single-turn anti-sycophancy

## 12. User preferences (DO NOT FORGET)
- Caveman mode ON (terse responses, drop articles/filler/pleasantries)
- HATES project name "Stakeholder Gym" → use "Meta" everywhere
- Wants real bold technical claims, NOT hype. Has called out over-hyping 3+ times.
- Numbers go in pitch ONLY after we measure them.
- Sleeps 3-hour increments; training overnight is expected
- Wants every commit pushed (no local-only work)
- Skip mentor brief / slide deck / HF Spaces unless directly asked
- "Reckless crazy" is the chosen ambition level (full Tier 1-5 scope)
- Don't add Co-Authored-By to commits

## 13. Architectural innovations (lead pitch with these)
1. **Trainable LLM adversary as part of env** — Qwen 3B LoRA with own DPO objective (maximize agent failure)
2. **Failure-mining → next-round training data** — round 1 mistakes become round 3 pairs (real closed-loop CoEvolve)
3. **MemoryUpdateAction** — MemAgent-style overwrite memory as 10th env action type, trained via DPO chosen examples
4. **MemoryAgentBench 4-competency scoring** — AR/TTL/LRU/CR aligned to ICLR 2026 paper
5. **Format reward + silent-cave penalty** in DPO data — cures cold-start WAIT-collapse
6. **LLM-driven scenario synthesis** via HF Inference Providers — endless adversarial scenarios from weakness profiles
7. **Multi-stakeholder coordination groups** — false-consensus + ally-pile-on collusion
8. **Multiplicative terminal reward** — `(2·achieved-1)·max(0,1-bad_ratio)` — unhackable
9. **SFT cold-start + DPO + Coevolution** stacked — DeepSeek-R1 + Tülu 3 + CoEvolve recipes combined

End of handoff. Dense + complete.
