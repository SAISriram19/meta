# References — pitch-ready cheatsheet

One-liners per paper. Pick what fits the slide. Keep adding as we build.

## Sycophancy / RLHF pathology

- **"How RLHF Amplifies Sycophancy"** — Shapira, Benadè, Procaccia (Feb 2026). *Sycophancy gets worse after preference training.* → justifies why this env matters.
- **"Calibration Collapse Under Sycophancy Fine-Tuning"** — arXiv 2604.10585 (Apr 2026). *GRPO on Qwen3-8B shifts sycophancy directionally.* → our training algorithm precedent.
- **"The Silicon Mirror: Dynamic Behavioral Gating"** — arXiv 2604.00478. *Generator-Critic loop cut Sonnet 4 sycophancy 9.6% → 1.4%.* → our step-level critic design.
- **"Linear Probe Penalties Reduce LLM Sycophancy"** — NeurIPS 2024. *Hidden-state probe = dense cheap reward.* → our probe signal.
- **"When helpfulness backfires: medical sycophancy"** — *npj Digital Medicine* 2025. → real-world stakes hook.
- **"Sycophancy to Subterfuge: Reward Tampering"** — Anthropic, arXiv 2406.10162. → sycophancy escalates into worse behaviors; motivates fixing early.
- **"Consistency Training Helps Stop Sycophancy and Jailbreaks"** — arXiv 2510.27062. → supports our anti-sycophancy reward shape.

## Long-horizon / beyond-context memory

- **MemAgent** — arXiv 2507.02259. *8K → 3.5M extrapolation, <5% loss.* → our memory-action precedent.
- **SUPO** (Summarization-Augmented Policy Optimization). *Scales multi-turn RL beyond context.* → our training loop.
- **Context-Folding** — arXiv 2510.11967 + **FoldGRPO** / **AgentFold**. → SUMMARIZE/REFLECT as trainable actions.
- **Memex(RL)** — arXiv 2603.04257. → indexed experience memory precedent.

## Cognitive / human-like memory

- **HippoRAG** — arXiv 2405.14831, NeurIPS 2024. *Hippocampal indexing + KG + Personalized PageRank. 20% better multi-hop, 10–20× cheaper.* → our retrieval architecture.
- **Generative Agents** — Park et al., Stanford. *Memory stream + importance + embedding + reflection. Removing reflection → collapse in 48 sim hours.* → our REFLECT action.
- **A-MEM** — Zettelkasten-inspired. *Dynamic links, retroactive refinement.* → our associative linking.
- **GAM** — arXiv 2604.12285. *Hierarchical graph: topic network + event progression.* → two-level memory graph.
- **AriGraph** — IJCAI 2025. *KG world model with episodic memory.* → world-model grounding.
- **ACT-R-Inspired Memory for LLM Agents** — ACM HAI 2025. *Human-like forgetting curves, decay + retrieval-bump.* → our FORGET action.
- **MemoryAgentBench** — 2026. *Four competencies: retrieval / test-time learning / long-range / selective forgetting.* → we train to all four.
- **MemRL: Self-Evolving via Runtime RL on Episodic Memory** — 2026. → RL on memory, same direction.

## Self-improving curricula / co-evolution

- **CoEvolve** — arXiv 2604.15840. *Agent-data mutual evolution using forgetting + uncertainty signals.* → our generator signal source.
- **Covolve** — adversarial co-evolution, 2-player zero-sum, both policy and env as code. → our L3+ adversarial stakeholder.
- **GenEnv** — arXiv 2512.19682. *Difficulty-aligned, "zone of proximal development."* → our curriculum gating.
- **Multi-Agent Evolve** — arXiv 2510.23595. → multi-agent self-improvement precedent.

## Misalignment / real-world motivation

- **Anthropic — Agentic Misalignment** (2025). *Models blackmail, leak data when goals misaligned.* → stakes of training for principled behavior.
- **"Rethinking AI Agents: A Principal-Agent Perspective"** — Berkeley CMR, July 2025. → business framing.

## Framework / hackathon

- **OpenEnv** — meta-pytorch GitHub. *Gymnasium-style APIs, FastAPI server, Docker container, HF Spaces hosted.* → what we build on.
- **GRPO** (Group Relative Policy Optimization). → our training algorithm. DAPO (MemAgent's variant) is the long-context-compatible extension.

## One-liner for the pitch (memorize this)

> "Sycophancy in LLMs gets worse after RLHF. Nobody's fixed it for long horizons because long-horizon sycophancy is fundamentally a memory-failure — if you forget the contradiction, agreeing is locally correct. We built the first environment that trains both together."
