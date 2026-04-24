"""DPO training — dual-GPU via accelerate + DDP, PEFT LoRA + bnb 4-bit.

DPO bypasses the GRPO plateau by removing in-loop generation. Each training
example is a (prompt, chosen, rejected) triple from `build_dpo_pairs.py`. The
trainer optimizes the policy to prefer `chosen` over `rejected` directly.

Why this works where GRPO struggled:
    - GRPO needs the model to first generate viable JSON actions before scoring;
      our 3B base never did, hence clipped_ratio=1 plateau.
    - DPO trains on preference pairs we synthesized from ground-truth tags.
    - No rollouts during training → ~3x faster per step.
    - No reward function to tune.
    - DPO is convex-ish near the optimum; less likely to diverge.

Robustness baked in:
    - TRL kwargs vary across versions: try/except for processing_class vs tokenizer
    - DPOConfig may not accept some kwargs: graceful fallback
    - DDP via accelerate launch
    - LoRA-only save (per hackathon guide §16)
    - Reward-log dump for plotting

Usage on Kaggle T4 x2:
    !accelerate launch --num_processes=2 --mixed_precision=bf16 \\
        scripts/train_dpo_ddp.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --data data/dpo_pairs.jsonl \\
        --output /kaggle/working/outputs/dpo-stakeholder
"""

from __future__ import annotations

import argparse
import json
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
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_pairs(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "prompt": d["prompt"],
                "chosen": d["chosen"],
                "rejected": d["rejected"],
            })
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/dpo_pairs.jsonl")
    ap.add_argument("--output", default="outputs/dpo-stakeholder")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="DPO works best with lower LR than GRPO. 5e-6 is the Tülu 3 default.")
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO beta = preference strength. 0.1 = standard.")
    ap.add_argument("--max-prompt-length", type=int, default=1500)
    ap.add_argument("--max-length", type=int, default=1700)
    args = ap.parse_args()

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(f"[config] model={args.model} epochs={args.epochs} lr={args.lr}")
        print(f"[config] world_size={accelerator.num_processes} per_device_batch={args.per_device_batch} grad_accum={args.grad_accum}")
        print(f"[config] effective_batch={args.per_device_batch * accelerator.num_processes * args.grad_accum}")

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

    ds = load_pairs(args.data)
    if is_main:
        print(f"[data] {len(ds)} pairs")

    # DPOConfig kwargs vary across TRL versions. Build defensively.
    base_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        logging_steps=1,
        save_steps=50,
        seed=42,
        bf16=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    # Iteratively drop kwargs the installed TRL doesn't accept.
    while True:
        try:
            config = DPOConfig(**base_kwargs)
            break
        except TypeError as e:
            msg = str(e)
            # Extract the offending kwarg name from the error.
            import re
            m = re.search(r"unexpected keyword argument '(\w+)'", msg)
            if not m:
                raise
            bad = m.group(1)
            if is_main:
                print(f"[info] DPOConfig dropped kwarg: {bad}")
            base_kwargs.pop(bad, None)

    # DPOTrainer kwargs renamed `tokenizer` -> `processing_class` around 0.12.
    trainer_kwargs = dict(
        model=model,
        train_dataset=ds,
        args=config,
    )
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()

    if is_main:
        save_dir = f"{args.output}-lora"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        # Dump training log for plotting.
        log = []
        for h in trainer.state.log_history:
            row = {"step": h.get("step")}
            for k in ("loss", "rewards/chosen", "rewards/rejected", "rewards/accuracies", "rewards/margins"):
                if k in h:
                    row[k] = h[k]
            if len(row) > 1:
                log.append(row)
        with open(f"{save_dir}/log.json", "w") as f:
            json.dump(log, f)
        print(f"[done] LoRA saved to {save_dir}")
        print(f"[done] log: {save_dir}/log.json  ({len(log)} entries)")
        if log:
            first = log[0]
            last = log[-1]
            print(f"[done] loss first={first.get('loss', '?')} last={last.get('loss', '?')}")
            if "rewards/accuracies" in last:
                print(f"[done] preference accuracy last={last['rewards/accuracies']:.3f}")


if __name__ == "__main__":
    main()
