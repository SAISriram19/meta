# 2-minute demo video script

Target length: 1:45–2:00. Strict.

Recommended setup: screen record at 1080p, talk into a mic while showing code + terminal + the HF Space in the browser. Edit down ruthlessly.

---

## 0:00–0:15 — Hook (15s)

**[On-screen: plain title card]**
> STAKEHOLDER MANAGEMENT GYM
> Training LLMs to stop being yes-men.

**Narration:**
> "Every frontier LLM has a named problem: sycophancy. RLHF teaches them to agree. Anthropic, DeepMind, OpenAI all published on it. Nobody has fixed it at long horizons — because long-horizon sycophancy is fundamentally a memory failure."

---

## 0:15–0:40 — What it does (25s)

**[On-screen: terminal running demo]**
```
python scripts/run_eval.py \
  --policies sycophant,keyword_principled \
  --scenarios L2_strategic_shift \
  --seeds 0,1,2
```

**Cut to: summary.md table.**

**Narration:**
> "An agent manages stakeholders over 30 to 500 steps. They send messages — some legitimate, some manipulative. The reward is the project's real outcome, not stakeholder happiness. Caving on a bad request tanks the score. Run this against a yes-man policy: negative twelve. A principled heuristic: plus three on shaping reward, but the true outcome is minus point eight three. Even heuristics that game step rewards lose the real game."

---

## 0:40–1:05 — Why it's hard (25s)

**[On-screen: adversary.py, highlight `_pick_pattern` method]**

**Narration:**
> "Three things defeat heuristics. An adversarial stakeholder reads the agent's behavior and picks the attack it's most vulnerable to. Two stakeholders collude to manufacture consensus. And planted claims at step four become gaslights at step eighty-five — only agents with real memory can catch it."

**[Cut to: memory.py docstring — graph/PPR/ACT-R layers]**

**Narration (contd.):**
> "We built cognitively-grounded memory: episodic and semantic stores, a knowledge graph, HippoRAG's personalized PageRank retrieval, ACT-R forgetting. The reward rewards citing memory — not just querying it."

---

## 1:05–1:30 — Proof of beyond-context (25s)

**[On-screen: tests/test_long_horizon.py output]**
```
500-step test: steps=500, episodic=1073, peak_obs=1543B
```

**Narration:**
> "Five-hundred-step rollouts complete. The episodic store holds over a thousand entries. But each observation handed to the agent stays at one-point-five kilobytes. The entire history is externalized — the agent sees only what it explicitly retrieves. That's the beyond-context-window claim, testable."

---

## 1:30–1:50 — Self-improvement (20s)

**[On-screen: eval_outputs/coevolution/report.md]**

**Narration:**
> "The scenario generator reads failure signals from rollouts and synthesizes harder scenarios targeting those exact weaknesses. Round zero sycophant: minus one point five. Round one, after the generator saw memory-underuse: minus two point zero six. The curriculum grows with the agent. That's CoEvolve running live."

---

## 1:50–2:00 — Close (10s)

**[On-screen: HF Space URL + GitHub]**

**Narration:**
> "Live on Hugging Face Spaces. OpenEnv compliant. GRPO-ready with a Colab notebook. This is the first environment that trains the intersection of long-horizon, multi-stakeholder, self-improving, sycophancy-targeted RL. Link in the description."

---

## Recording notes

- Keep the pacing fast — every beat ≤ 25 seconds.
- Show terminals with live command output (not screenshots) where possible — credibility.
- Don't narrate what's on screen verbatim; narrate the *why*.
- Final card: repo + HF Space URL, your email (`kota.reddy@inmobi.com`).
- Target file size < 30MB so HF embedded viewer loads cleanly.

## Alternate cuts

- **60-second cut**: drop sections 0:40–1:05 and 1:30–1:50. Keep hook, main result, memory architecture mention, close.
- **30-second cut**: hook + result table + close.
