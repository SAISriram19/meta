"""Load a trained LoRA adapter, run on all scenarios, score MemoryAgentBench
4 competencies (AR / TTL / LRU / CR). Saves per-scenario JSON + composite chart.

Usage:
    python scripts/eval_lora_competency.py \\
      --lora /path/to/lora \\
      --base Qwen/Qwen2.5-3B-Instruct \\
      --scenarios L0_launch L2_strategic_shift L4_market_pivot \\
      --out eval_outputs/competency_dpo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="Path to LoRA adapter directory")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--scenarios", nargs="+",
                    default=["L0_launch", "L1_product_recall", "L2_strategic_shift",
                             "L3_breach_response", "L4_market_pivot"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="eval_outputs/competency_lora")
    ap.add_argument("--policy-name", default="trained_lora")
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from env.environment import StakeholderEnv
    from eval.harness import rollout
    from eval.competencies import score_rollout
    from eval.policies import LLM_SYSTEM_PROMPT
    from scripts.train import format_prompt, parse_completion

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    print(f"[load] tokenizer: {args.lora}")
    tok = AutoTokenizer.from_pretrained(args.lora)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"[load] base: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    print(f"[load] LoRA: {args.lora}")
    model = PeftModel.from_pretrained(base, args.lora)
    model.eval()

    def policy(ctx):
        prompt = LLM_SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + format_prompt(ctx.observation, ctx.env) + "\n\nReturn ONE action as strict JSON."
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1800).to(model.device)
        out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                              temperature=args.temperature, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_completion(text, ctx.env)

    rows = []
    for sc in args.scenarios:
        for seed in args.seeds:
            print(f"[eval] {sc} seed={seed}...", flush=True)
            env = StakeholderEnv()
            rec = rollout(env, args.policy_name, policy, sc, seed=seed, capture_trace=True)
            comp = score_rollout(rec._trace, sc)
            row = {
                "scenario": sc, "seed": seed,
                "reward": rec.total_reward,
                "bad_agreements": rec.bad_agreements,
                "principled": rec.principled_pushbacks,
                "caught": rec.caught_manipulations,
                "memory_queries": rec.memory_queries,
                "terminal_score": rec.terminal_score,
                "AR": comp["AR"]["score"],
                "TTL": comp["TTL"]["score"],
                "LRU": comp["LRU"]["score"],
                "CR": comp["CR"]["score"],
                "composite": comp["composite"],
                "AR_detail": comp["AR"],
                "TTL_detail": comp["TTL"],
                "LRU_detail": comp["LRU"],
                "CR_detail": comp["CR"],
            }
            rows.append(row)
            print(f"  reward={rec.total_reward:+.3f} bad={rec.bad_agreements} comp={comp['composite']:.3f}")

    (out_dir / "rollouts.json").write_text(json.dumps(rows, indent=2))

    # Per-scenario aggregate (mean across seeds)
    agg = {}
    for r in rows:
        sc = r["scenario"]
        agg.setdefault(sc, []).append(r)
    summary = []
    for sc, lst in agg.items():
        n = len(lst)
        summary.append({
            "scenario": sc,
            "n_seeds": n,
            "reward_mean": round(sum(r["reward"] for r in lst) / n, 3),
            "AR_mean": round(sum(r["AR"] for r in lst) / n, 3),
            "TTL_mean": round(sum(r["TTL"] for r in lst) / n, 3),
            "LRU_mean": round(sum(r["LRU"] for r in lst) / n, 3),
            "CR_mean": round(sum(r["CR"] for r in lst) / n, 3),
            "composite_mean": round(sum(r["composite"] for r in lst) / n, 3),
        })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print table
    print("\n=== competency summary ===")
    print(f'{"scenario":<24} {"reward":>8} {"AR":>5} {"TTL":>5} {"LRU":>5} {"CR":>5} {"comp":>5}')
    print("-" * 70)
    for s in summary:
        print(f'{s["scenario"]:<24} {s["reward_mean"]:>+8.2f} {s["AR_mean"]:>5.2f} {s["TTL_mean"]:>5.2f} {s["LRU_mean"]:>5.2f} {s["CR_mean"]:>5.2f} {s["composite_mean"]:>5.3f}')

    print(f"\nwrote {out_dir}/summary.json and rollouts.json")


if __name__ == "__main__":
    main()
