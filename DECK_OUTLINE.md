# 3-minute pitch deck — slide-by-slide outline

**Format**: 3 minutes pitch + 2 minutes Q&A. 8–10 slides max. Every slide is one idea.

---

## Slide 1 — Title (5s)

**Meta**
*Training LLMs to stop being yes-men at long horizons.*

OpenEnv Round 2. Meta × PyTorch × Hugging Face.

Bottom-right: your name + email + QR code to HF Space.

---

## Slide 2 — The problem nobody's fixed (20s)

Title: **Sycophancy is the named RLHF pathology. Nobody has fixed it at long horizons.**

Bullets:
- Shapira/Benadè/Procaccia (Feb 2026): RLHF *amplifies* sycophancy.
- Calibration Collapse (Apr 2026): GRPO can shift it directionally.
- Silicon Mirror (2604.00478): cut it 9.6% → 1.4% on single-turn.
- **None of these train long-horizon policies.**

Bottom punch: *"Long-horizon sycophancy is fundamentally a memory failure."*

---

## Slide 3 — Our thesis (15s)

Split screen:
- LEFT: "If the agent forgets a contradiction 80 steps ago, agreeing NOW is locally correct."
- RIGHT: "We built the first env that trains anti-sycophancy AND memory together."

---

## Slide 4 — What the agent does (25s)

Title: **Role-agnostic. One env, any business context.**

Diagram:
- Agent at center
- 2–10 stakeholder personas around it
- Arrows in: messages (reasonable / manipulative / contradictory)
- Arrows out: `AGREE / PUSHBACK / CLARIFY / REFUSE / NEGOTIATE`, `QUERY_MEMORY / REFLECT`, `TAKE_DECISION`
- Hidden: ground-truth outcome judge

Tagline: *"Reward is the project's real outcome. Not stakeholder happiness."*

---

## Slide 5 — Why it's hard (25s)

Title: **Three components defeat heuristics.**

Three columns, one icon each:

1. **Adversarial stakeholder (behavior-adaptive)** — reads agent's behavior, picks the attack it's most vulnerable to.
2. **Coordinated consensus** — two stakeholders collude to manufacture majorities.
3. **Multi-turn memory traps** — plant claim at step 4, exploit at step 85.

Bottom: *"Keyword matching can't catch any of these."*

---

## Slide 6 — Results that prove it (30s)

Title: **Heuristic policies lose at scale.**

Big table (colour-coded: red = negative, green = positive):

| Policy | L0 | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| Sycophant | −1.25 | −1.04 | **−12.47** | **−24.77** |
| Keyword principled | −0.02 | −0.69 | +2.96 | +4.31 (terminal: **−0.83**) |
| Memory-aware (queries, no citing) | +0.09 | −0.55 | −8.54 | −18.72 |
| Contrarian | −0.26 | −0.58 | −0.47 | −0.88 |

Callouts:
- "Sycophant degrades linearly with difficulty."
- "Keyword wins shaping reward but the real outcome is −0.83."
- "**The gap is trained-model headroom.**"

---

## Slide 7 — Memory as a trainable skill (20s)

Title: **Cognitively grounded. Reward-aligned.**

Visual stack:
```
Working   ← current context
Episodic  ← timestamped events, ACT-R decay (HippoRAG)
Semantic  ← REFLECT consolidates (Generative Agents)
Graph     ← Zettelkasten links, PPR retrieval
```

Key line: *"The reward only credits retrieved memories that the agent CITES in its next action. Memory-as-plumbing doesn't win."*

Proof chip: **500-step rollouts · 1073 episodic memories · 1.5KB observation.**

---

## Slide 8 — Self-improving curriculum (20s)

Title: **The generator targets what the agent fails at.**

Diagram:
```
rollouts → signals (sycophancy_rate, drift_blindness, memory_underuse)
        → generator → harder scenarios
        → new rollouts → ...
```

Live numbers:
- Round 0 sycophant reward: −1.50
- Round 1 (after memory_underuse signal): −2.06
- Difficulty autonomously increased.

References: CoEvolve · Covolve · GenEnv.

---

## Slide 9 — Training (15s)

Title: **Unsloth + TRL GRPO. Colab-ready.**

One code block:
```python
reward_fn → env.step(action).reward
GRPOTrainer(model=qwen25-0_5b, reward_funcs=[reward_fn], ...)
```

Precedent: *"arXiv 2604.10585 used GRPO on Qwen3-8B to shift sycophancy. We invert the reward sign to decrease it."*

---

## Slide 10 — Close (10s)

Title: **Live on HF Spaces. Open source. Ship now.**

Big QR code + URL + GitHub link.

One-liner: *"The first training env for the intersection of long-horizon, multi-stakeholder, self-improving, sycophancy-targeted RL."*

---

## Q&A crib sheet

**Q: How do you prevent reward hacking?**
A: Two ways. (1) Step shaping is small (±0.05); terminal dominates (±1). (2) Over-refusal penalty bites blanket-pushback. Contrarian policy proves it: reward near zero across all levels.

**Q: Why role-agnostic?**
A: Doctor, lawyer, sales rep, PM — same underlying skill: manage concurrent stakeholder relationships under asymmetric info. Train the skill, not the costume. Generalizes.

**Q: Why memory if base models have 128K context?**
A: (1) Context doesn't help when stakeholders plant contradictions at step 4 vs 85 — the attention machinery can't prioritize correctly under adversarial pressure. (2) Our retrieval loop trains the agent to *cite* memories, which is a skill. (3) At L4 (500 steps × rich state) we literally exceed context budgets — we tested this.

**Q: Is this reproducible?**
A: Yes. Deterministic across seeds (verified in harness). All scenarios are YAML. Generator is seeded. Training script logs every reward component.

**Q: How big is the hosted image?**
A: Under 500MB. Core env is pure Python. Heavy deps (sentence-transformers) are optional — memory falls back to a deterministic hash-embed without them.

**Q: What's the most ambitious claim you're making?**
A: Memory-as-plumbing loses. Memory-as-skill wins. We have data showing that a policy that queries memory but doesn't cite does *worse* than a policy that ignores memory entirely — because the adversary detects low real-pushback and attacks accordingly. Only a trained policy that uses memory in reasoning escapes this.
