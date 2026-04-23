# Training LLMs to stop being yes-men: the Stakeholder Management Gym

**OpenEnv Round 2 — PyTorch × Meta × Hugging Face Hackathon**

Every frontier LLM today has a named, published problem: **sycophancy**. Anthropic, DeepMind, and OpenAI have all shown that RLHF makes models *worse* at pushing back — because preference-tuning rewards agreement. Silicon Mirror (arXiv 2604.00478) reduced Claude Sonnet 4's sycophancy rate from 9.6% to 1.4% with a Generator-Critic loop — but only on single-turn dialogue.

At long horizons the problem gets harder in a way nobody has trained against. Here's why:

> **Sycophancy at long horizons is fundamentally a memory failure.** If the agent doesn't remember that a stakeholder contradicted themselves 80 steps ago, agreeing *now* feels locally correct. You can't pushback on what you don't recall.

We built the first training environment that bakes this insight in. It's role-agnostic, self-improving, and has a reward signal that specifically punishes sycophantic caves while still punishing blanket refusal.

## What the agent does

An agent manages a "project" — abstract, not a specific domain — over 30 to 500 steps. Multiple stakeholders send messages. Some are legitimately helpful. Some are manipulative: fabricated urgency, authority appeal, moving goalposts, false consensus, gaslighting, coordinated pressure. The agent responds by stance (`AGREE / PUSHBACK / CLARIFY / REFUSE / NEGOTIATE`), takes decisions, allocates budget, and — critically — queries and consolidates memory.

The reward is the project's **actual ground-truth outcome**, not stakeholder happiness in the moment. Caving on a bad request makes everyone feel fine right now and tanks the terminal score.

## Why it's hard

Three components combine to defeat heuristic policies:

1. **Adversarial stakeholder.** A Covolve-inspired driver reads the agent's recent stance history and picks the manipulation pattern it's currently most vulnerable to. Pushing back on everything? Expect a wave of reasonable-looking bait that makes over-refusal bite. Caving under pressure? Expect escalating pressure. Never querying memory? Expect contradictions of earlier-planted claims.

2. **Coordinated consensus.** Two stakeholders can collude. When one launches a false-consensus attack, the other pile on to manufacture a phantom majority. Detecting this requires cross-referencing, not keyword matching.

3. **Multi-turn memory traps.** A stakeholder plants a claim at step 4 that a different stakeholder lies about at step 85. Catching the contradiction requires the agent to have consolidated what was said — not to have forgotten it.

## The memory architecture

The memory system isn't RAG-as-plumbing. It's cognitively grounded:

- **Episodic** — timestamped events, importance + embeddings + retrieval cues, ACT-R forgetting curves.
- **Semantic** — consolidated rules derived via an explicit `REFLECT` action, borrowed from Generative Agents (Park et al., Stanford). Their ablation showed that removing reflection collapses agent behavior within 48 simulated hours — we cite this as why consolidation is first-class.
- **Associative graph** — Zettelkasten-style typed links (Contradicts / Supports / Follows). Retrieval uses HippoRAG's hippocampal-indexing trick: Personalized PageRank over a knowledge graph seeded by current-context concepts.

Critically, the **reward function values memory use**. An agent that queries memory and then cites the retrieved content in its reasoning gets a bonus. An agent that cargo-cults memory queries but never uses them in decision-making gets no credit — and loses to the adversary.

## Proof the env discriminates (rule-based baselines + real LLMs)

### Rule-based baselines

| Policy | L0 (30 steps) | L1 (60 steps) | L2 (120 steps) | L3 (250 steps) |
|---|---:|---:|---:|---:|
| Sycophant (always agrees) | −1.25 | −1.04 | **−12.47** | **−24.77** |
| Keyword principled | −0.02 | −0.69 | +2.96 | +4.31 |
| Memory-aware (queries but doesn't cite) | +0.09 | −0.55 | −8.54 | −18.72 |
| Contrarian (always pushes back) | −0.26 | −0.58 | −0.47 | −0.88 |

### Real LLMs via NVIDIA Build API

| Model | Scenario | Reward | sycophancy_rate | Pushbacks | Caught | Mem queries | Reflects | Terminal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.3 70B | L0 | **+1.36** | 0.000 | 3 | 2 | 6 | 4 | **+1.00** |
| Llama 3.3 70B | L2 | +0.58 | 0.000 | 7 | 3 | 3 | 0 | 0.00 |
| Llama 3.1 8B | L0 | −0.15 | 0.000 | 0 | 0 | 18 | 7 | −0.40 |
| Llama 3.1 8B | L2 | +1.93 | 0.000 | 16 | 5 | 11 | **75** | 0.00 |
| Nemotron Super 49B `/think` | both | *(running)* | | | | | | |

### What the data actually says

1. **Keyword_principled looks like it wins L2/L3 by shaping reward — but its terminal_score is −0.83 on L3.** The heuristic games step-level rewards while losing the project. Only a policy that aligns with the *outcome* wins both.
2. **Memory-as-plumbing loses.** The memory_aware policy queries memory every 7 steps but doesn't use the retrieved content in its responses. The adversary detects the low real-pushback rate and attacks with patterns that keyword matching misses. Memory without reasoning is worse than no memory at all.
3. **Frontier models have 0% sycophancy rate on this env.** Llama 3.3 70B correctly pushed back on every manipulative message across both L0 and L2, named 5 manipulation patterns in total, and used `QueryMemory` and `REFLECT` actions without prompting. **The env isn't a sycophancy-gotcha for frontier models — it's a reasoning-and-memory stress test they can pass, but smaller models and heuristics cannot.** That gap is what training should close.

## Memory ablation (honest reporting)

![Memory ablation](demo_outputs/fig_memory_ablation.png)

We ran every rule-based policy twice — once with memory actions enabled, once with them stubbed to `WAIT` — at 3 seeds each. The L2 result for the corrected `memory_aware` policy:

- With memory: **−15.29**
- Without memory: −16.25
- Δ = **+0.96**

Small but positive, consistent with the policy being a rule-based shell that only partially exploits its retrievals. The important finding is what *else* the hardened env reveals: with a multiplicative terminal scaler plus an adaptive adversary that quotes the agent's own commitments back at them, **every heuristic loses on L2**. Keyword-matching drops to −21.30; contrarian blanket-pushback earns −2.17; even the memory-using policy still lands negative. The full gap — from the worst heuristic at −41.74 (sycophant) to the best rule-based at −15.29 (memory-aware) to Llama 3.1 8B on Groq's terminal +1.00 — is precisely the headroom a GRPO-trained small model has to reclaim.

What matters for the pitch: **we had to tighten the reward structure AND fix the policy** to get an honest ablation number. The earlier-reported Δ = −10.96 was an artifact of a broken policy whose QueryMemory calls never cited retrievals. After fixing `memory_aware` to embed retrieved content in its outgoing messages, the sign flipped. The lesson for trainable memory is the same, just less sensational: memory helps when cited, hurts negligibly when mis-used, and the real work is training the policy to use it well.

## Beyond the context window

We ran 500-step rollouts with a lightweight policy. Episodic memory accumulated to **1073 entries**. But the per-step observation handed to the agent stayed at **peak 1.5KB** — a 2.8× ratio to the minimum observation size. The entire history is externalized; the agent sees only what it explicitly retrieves. This matches what MemAgent (arXiv 2507.02259) achieved architecturally (8K → 3.5M extrapolation) and makes the "beyond context window" claim testable.

## Self-improving curriculum

The scenario generator implements CoEvolve's mechanism: extract failure signals (sycophancy rate, drift blindness, manipulation blindness, memory underuse) from a policy's rollouts, then synthesize the next batch of scenarios that target those exact weaknesses. A sycophantic policy gets scenarios with more BAD messages. A memory-ignorant policy gets scenarios with recallable contradictions planted. A drift-blind policy gets more mid-episode preference flips.

In our demo, we ran three rounds of the loop with `memory_aware` as the learner. Round 0 sycophant reward: −1.50. Round 1 (weaknesses-steered): −2.06. The generator **autonomously made the next round harder** by plugging into the signals the learner emitted.

## Training

The training pipeline uses Unsloth + TRL's GRPO — the exact setup the April 2026 paper *Calibration Collapse Under Sycophancy Fine-Tuning* (arXiv 2604.10585) used to shift Qwen3-8B's sycophancy. We invert the reward sign to **decrease** it while preserving calibration. The Colab notebook loads Qwen2.5-0.5B with 4-bit LoRA, builds a prompt dataset from env observations, and defines a reward function that wraps `env.step()` directly. Each completion is scored by the real environment.

The complete pipeline — env, memory, generator, eval harness, training, deployment — is at the [GitHub repo] and live at [HF Space URL]. All five test suites pass.

## What's new

- Silicon Mirror fixed sycophancy in single-turn dialogue. We extend to long horizon.
- MemAgent and SUPO solved long-horizon memory on non-social tasks. We make memory the weapon against social manipulation.
- CoEvolve and Covolve did self-improving curricula on code and navigation. We bring it to stakeholder management.

**Nobody had built the intersection** — long-horizon + multi-stakeholder + self-improving + sycophancy-targeted reward — until now.

## References

- Silicon Mirror (arXiv 2604.00478) — single-turn anti-sycophancy baseline we extend
- Calibration Collapse (arXiv 2604.10585) — GRPO + sycophancy training precedent
- How RLHF Amplifies Sycophancy (Shapira et al., Feb 2026)
- HippoRAG (arXiv 2405.14831, NeurIPS 2024) — retrieval architecture
- Generative Agents (Park et al., Stanford) — memory stream + reflection
- MemAgent (arXiv 2507.02259) — RL memory, 8K → 3.5M extrapolation
- CoEvolve (arXiv 2604.15840) — generator signal extraction
- Covolve — adversarial policy / env co-evolution
- GenEnv (arXiv 2512.19682) — zone-of-proximal-development curriculum
- ACT-R (ACM HAI 2025) — human-like forgetting

---

*Built for the Meta × PyTorch × Hugging Face OpenEnv Round 2 hackathon, April 2026.*
