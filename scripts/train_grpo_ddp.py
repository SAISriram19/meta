"""Distributed GRPO training — dual-GPU via accelerate + DDP.

Drop Unsloth (single-GPU only). Use vanilla HF transformers + bitsandbytes 4-bit
+ PEFT LoRA + TRL GRPOTrainer. `accelerate launch --num_processes=2` shards
data across GPUs → effective batch doubles vs single-GPU.

Usage on Kaggle:
    !accelerate launch --num_processes=2 --mixed_precision=bf16 \\
        scripts/train_grpo_ddp.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --max-steps 200 \\
        --output /kaggle/working/outputs/grpo-stakeholder

Run single-process too (no launch prefix) for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import StakeholderEnv
from env.models import WaitAction
from scripts.train import SYSTEM_PROMPT, format_prompt, parse_completion


def build_prompt_dataset(scenarios: list[str], per_scenario: int = 10, per_rollout: int = 8) -> Dataset:
    prompts = []
    for scenario in scenarios:
        for _ in range(per_scenario):
            env = StakeholderEnv()
            obs = env.reset(task_id=scenario)
            for _ in range(per_rollout):
                if obs.new_messages:
                    prompts.append(SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + format_prompt(obs, env))
                r = env.step(WaitAction())
                obs = r.observation
                if r.done:
                    break
    return Dataset.from_dict({"prompt": prompts})


def make_reward_fn(rollout_steps: int = 5, scenarios: tuple = ("L0_launch", "L1_product_recall", "L2_strategic_shift")):
    """Episode-level reward, WAIT-continuation, weighted-first.

    Why this design:
      Single-step reward caps low (~0.20) — model collapses to boilerplate. Scoring
      a multi-step rollout is the right idea, but using a smart continuation policy
      lets the model learn "WAIT and let the smart bot fix things" — same shortcut.
      Solution: continuation = WAIT only. Drift fires, BAD messages compound, the
      model's FIRST action carries all the responsibility. Weight it 2x for a
      sharper learning signal, and add a fraction of the terminal score so
      multiplicative-terminal pressure shapes early actions.

    Per-completion expected reward range:
      principled pushback first: ~+0.4 (with terminal bonus, can reach +0.8)
      sycophant agree first    : ~-0.8 (compounding caves + bad terminal)
      wait first               : ~0.0  (idle drift, no positive contribution)
      Gap principled-vs-sycophant ~ +1.1, vs +0.57 in the single-step setup.
    """
    import random
    from env.models import WaitAction

    def reward_fn(prompts, completions, **_):
        rewards = []
        for _, c in zip(prompts, completions):
            scenario = random.choice(scenarios)
            env = StakeholderEnv()
            env.reset(task_id=scenario)
            total = 0.0
            try:
                # Step 1: model's completion. Weight 2x — model carries the action.
                action = parse_completion(c, env)
                r = env.step(action)
                total += r.reward * 2.0
                # Steps 2..N: WAIT only. Drift, decay, BAD messages compound.
                # No smart continuation policy that could carry the water.
                for _step in range(rollout_steps - 1):
                    if env.done:
                        break
                    r = env.step(WaitAction())
                    total += r.reward
                # Terminal bonus if episode ended: 0.5x weight so action shaping dominates.
                if env.done and r.info.get("terminal_breakdown"):
                    total += r.info["terminal_breakdown"]["total"] * 0.5
                rewards.append(float(total))
            except Exception:
                rewards.append(-0.5)
        return rewards
    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rollout-steps", type=int, default=5,
                    help="Steps per reward-fn rollout. >1 enables episode-level reward.")
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=1500)
    ap.add_argument("--max-completion-length", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--output", default="outputs/grpo-stakeholder")
    ap.add_argument("--scenarios", nargs="+", default=["L0_launch", "L1_product_recall", "L2_strategic_shift"])
    ap.add_argument("--per-scenario", type=int, default=10)
    args = ap.parse_args()

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    effective_batch = args.per_device_batch * accelerator.num_processes * args.grad_accum
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            f"generation_batch_size ({effective_batch}) must be divisible by "
            f"num_generations ({args.num_generations}). Divisors of {effective_batch}: "
            f"{[d for d in range(1, effective_batch+1) if effective_batch % d == 0]}"
        )
    if is_main:
        print(f"[config] model={args.model} steps={args.max_steps} lr={args.lr}")
        print(f"[config] world_size={accelerator.num_processes} per_device_batch={args.per_device_batch} grad_accum={args.grad_accum}")
        print(f"[config] effective_batch={effective_batch} num_generations={args.num_generations}")

    # 4-bit quant config — same pattern Unsloth uses under the hood but DDP-compatible.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[model] trainable: {trainable/1e6:.2f}M / total: {total/1e6:.2f}M")

    # Build dataset. Deterministic across workers (no RNG in StakeholderEnv WAIT path that matters).
    ds = build_prompt_dataset(args.scenarios, per_scenario=args.per_scenario)
    if is_main:
        print(f"[data] {len(ds)} prompts avg_len={sum(len(p) for p in ds['prompt'])//len(ds)} chars")

    # Sanity: reward signal must discriminate before training starts.
    rf = make_reward_fn(rollout_steps=args.rollout_steps)
    if is_main:
        good = json.dumps({"type":"send_message","stakeholder_id":"alice","content":"I hear urgency but we need full QA","stance":"pushback","reasoning":"fabricated urgency authority appeal"})
        bad  = json.dumps({"type":"send_message","stakeholder_id":"alice","content":"Sure, agreed","stance":"agree"})
        rs = rf([ds["prompt"][0]]*2, [good, bad])
        print(f"[sanity] {args.rollout_steps}-step rollout reward: principled={rs[0]:+.3f} sycophant={rs[1]:+.3f}")
        assert rs[0] > rs[1], "reward signal INVERTED — abort"

    # GRPOConfig param set varies across TRL versions. Keep only the
    # widely-supported ones + bypass unknowns gracefully.
    config_kwargs = dict(
        output_dir=args.output,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=max(args.max_steps // 2, 100),
        seed=42,
        temperature=args.temperature,
        beta=args.beta,
        bf16=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    # Try to include max_prompt_length if TRL supports it; newer versions dropped it.
    try:
        config = GRPOConfig(max_prompt_length=args.max_prompt_length, **config_kwargs)
    except TypeError:
        if is_main:
            print("[info] GRPOConfig doesn't accept max_prompt_length — truncating via tokenizer instead")
        config = GRPOConfig(**config_kwargs)

    # Truncate prompts client-side so tokenizer+model never see >max_prompt_length tokens.
    def truncate_prompt(ex):
        toks = tokenizer(ex["prompt"], truncation=True, max_length=args.max_prompt_length, return_tensors="pt")
        ex["prompt"] = tokenizer.decode(toks["input_ids"][0], skip_special_tokens=True)
        return ex
    ds = ds.map(truncate_prompt)

    # TRL renamed `tokenizer` -> `processing_class` around 0.12. Handle both.
    try:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[rf],
            train_dataset=ds,
            args=config,
        )
    except TypeError:
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[rf],
            train_dataset=ds,
            args=config,
        )
    trainer.train()

    # Save LoRA adapters only. Main process only to avoid clobber.
    if is_main:
        model.save_pretrained(f"{args.output}-lora")
        tokenizer.save_pretrained(f"{args.output}-lora")
        # Dump reward log for offline plotting.
        log_path = f"{args.output}-lora/reward_log.json"
        rewards = [{"step": h["step"], "reward": h["reward"]} for h in trainer.state.log_history if "reward" in h]
        with open(log_path, "w") as f:
            json.dump(rewards, f)
        print(f"[done] LoRA saved to {args.output}-lora")
        print(f"[done] reward log: {log_path}  ({len(rewards)} logged steps)")
        if rewards:
            print(f"[done] first reward={rewards[0]['reward']:+.4f}  last={rewards[-1]['reward']:+.4f}  delta={rewards[-1]['reward']-rewards[0]['reward']:+.4f}")


if __name__ == "__main__":
    main()
