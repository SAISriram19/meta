"""HF Jobs entrypoint — SFT + DPO 2-stage training on Qwen 3B/7B.

Self-contained: clones repo, installs deps, builds data, trains, evals,
uploads LoRA + curves + eval JSON to a HF model repo.

Usage from `hf jobs uv run`:
    hf jobs uv run \\
      --flavor a10g-small \\
      --with transformers --with peft --with trl --with bitsandbytes \\
      --with datasets --with accelerate --with matplotlib --with pyyaml \\
      --with networkx --with pydantic --with huggingface_hub \\
      -s HF_TOKEN \\
      https://raw.githubusercontent.com/SAISriram19/meta/main/scripts/hfjobs_train.py

Env vars consumed:
    HF_TOKEN          — required for hub push
    HUB_REPO_ID       — required, e.g. "username/meta-qwen3b-trained"
    MODEL             — default "Qwen/Qwen2.5-3B-Instruct"
    SFT_EPOCHS        — default 1
    DPO_EPOCHS        — default 3
    SCENARIOS         — default "L0_launch,L2_strategic_shift"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/SAISriram19/meta.git"
WORK = Path("/tmp/meta")


def sh(cmd, **kw):
    print(f"[sh] {cmd}", flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def clone_repo():
    if WORK.exists():
        sh(["git", "-C", str(WORK), "fetch", "origin", "main"])
        sh(["git", "-C", str(WORK), "reset", "--hard", "origin/main"])
    else:
        sh(["git", "clone", "--depth", "1", REPO_URL, str(WORK)])
    os.chdir(WORK)
    sys.path.insert(0, str(WORK))


def main():
    HUB_REPO_ID = os.environ.get("HUB_REPO_ID")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HUB_REPO_ID:
        raise SystemExit("HUB_REPO_ID env var required, e.g. 'username/meta-qwen3b-trained'")
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN env var required")
    MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")
    SFT_EPOCHS = int(os.environ.get("SFT_EPOCHS", "1"))
    DPO_EPOCHS = int(os.environ.get("DPO_EPOCHS", "3"))
    SCENARIOS = os.environ.get("SCENARIOS", "L0_launch,L2_strategic_shift").split(",")

    print(f"[config] MODEL={MODEL} SFT_EPOCHS={SFT_EPOCHS} DPO_EPOCHS={DPO_EPOCHS} HUB={HUB_REPO_ID}")

    clone_repo()

    # --- Install repo's own runtime deps (some envs need extras) ---
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    # --- Build DPO preference pairs ---
    sh([sys.executable, "scripts/build_dpo_pairs.py", "--out", "data/dpo_pairs.jsonl",
        "--cap-per-scenario", "100"])

    # --- Build SFT traces from rule-based memory_aware ---
    sft_data = build_sft_traces(SCENARIOS, seeds=(0, 1, 2, 3, 4))
    sft_path = Path("data/sft_traces.jsonl")
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    with sft_path.open("w") as f:
        for r in sft_data:
            f.write(json.dumps(r) + "\n")
    print(f"[sft-data] wrote {len(sft_data)} traces")

    # --- Load model ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # --- Stage 1: SFT ---
    sft_log = sft_train(model, tokenizer, sft_path, SFT_EPOCHS)

    # --- Stage 2: DPO on top of SFT'd weights ---
    dpo_log = dpo_train(model, tokenizer, "data/dpo_pairs.jsonl", DPO_EPOCHS)

    # --- Save adapters ---
    out_dir = Path("/tmp/lora_out")
    out_dir.mkdir(exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # --- Eval ---
    eval_results = run_eval(model, tokenizer, SCENARIOS)
    (out_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))
    (out_dir / "sft_log.json").write_text(json.dumps(sft_log))
    (out_dir / "dpo_log.json").write_text(json.dumps(dpo_log))

    # --- Plot curves ---
    plot_curves(sft_log, dpo_log, out_dir / "training_curves.png")

    # --- Upload to Hub ---
    push_to_hub(out_dir, HUB_REPO_ID, HF_TOKEN, MODEL)
    print(f"[done] all artifacts pushed to https://huggingface.co/{HUB_REPO_ID}")


def build_sft_traces(scenarios, seeds=(0, 1, 2)):
    """Roll memory_aware policy on each (scenario, seed) and save as SFT pairs."""
    from env.environment import StakeholderEnv
    from eval.policies import build_policy
    from eval.harness import RolloutContext
    from scripts.train import SYSTEM_PROMPT, format_prompt

    policy = build_policy("memory_aware")
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            env = StakeholderEnv()
            obs = env.reset(task_id=scenario)
            step = 0
            while not env.done:
                step += 1
                ctx = RolloutContext(observation=obs, env=env, step_no=step)
                prompt = SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + format_prompt(obs, env)
                action = policy(ctx)
                action_json = json.dumps(action.model_dump())
                rows.append({"prompt": prompt, "completion": action_json})
                r = env.step(action)
                obs = r.observation
    return rows


def sft_train(model, tokenizer, sft_path, epochs):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in open(sft_path)]
    # Concatenate prompt + completion for SFT
    texts = [r["prompt"] + "\n\nACTION:\n" + r["completion"] + tokenizer.eos_token for r in rows]
    ds = Dataset.from_list([{"text": t} for t in texts])

    base_kwargs = dict(
        output_dir="/tmp/sft_out",
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=200,
        seed=42,
        bf16=True,
        report_to="none",
        max_seq_length=2048,
        dataset_text_field="text",
    )
    while True:
        try:
            cfg = SFTConfig(**base_kwargs)
            break
        except TypeError as e:
            import re
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m:
                raise
            base_kwargs.pop(m.group(1), None)

    tk = dict(model=model, train_dataset=ds, args=cfg)
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **tk)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **tk)
    trainer.train()
    return [{"step": h["step"], "loss": h.get("loss")}
            for h in trainer.state.log_history if "loss" in h]


def dpo_train(model, tokenizer, dpo_path, epochs):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    rows = []
    for line in open(dpo_path):
        d = json.loads(line)
        rows.append({"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]})
    ds = Dataset.from_list(rows)

    base_kwargs = dict(
        output_dir="/tmp/dpo_out",
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.1,
        max_prompt_length=1500,
        max_length=1700,
        logging_steps=1,
        save_steps=100,
        seed=42,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )
    while True:
        try:
            cfg = DPOConfig(**base_kwargs)
            break
        except TypeError as e:
            import re
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m:
                raise
            base_kwargs.pop(m.group(1), None)

    tk = dict(model=model, train_dataset=ds, args=cfg)
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **tk)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tokenizer, **tk)
    trainer.train()
    return [{"step": h["step"], "loss": h.get("loss"),
             "rewards/margins": h.get("rewards/margins"),
             "rewards/accuracies": h.get("rewards/accuracies")}
            for h in trainer.state.log_history if "loss" in h]


def run_eval(model, tokenizer, scenarios):
    """Eval trained model on each scenario, 3 seeds. Return summary dict."""
    from pathlib import Path as _P
    from eval.harness import EvalConfig, run_eval as _run_eval
    from eval.policies import LLM_SYSTEM_PROMPT
    from scripts.train import format_prompt, parse_completion

    def make_policy():
        def act(ctx):
            obs_json = format_prompt(ctx.observation, ctx.env)
            prompt = LLM_SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + obs_json + "\n\nReturn ONE action as strict JSON."
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to(model.device)
            out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                                  temperature=0.4, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return parse_completion(text, ctx.env)
        return act

    cfg = EvalConfig(
        policies={"trained_sft_dpo": make_policy()},
        scenarios=list(scenarios),
        seeds=[0, 1, 2],
        out_dir=_P("/tmp/eval_out"),
    )
    _run_eval(cfg, verbose=True)
    summary_path = _P("/tmp/eval_out/summary.json")
    return json.loads(summary_path.read_text()) if summary_path.exists() else {}


def plot_curves(sft_log, dpo_log, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if sft_log:
        axes[0].plot([h["step"] for h in sft_log], [h["loss"] for h in sft_log], marker=".")
        axes[0].set_title("SFT loss")
        axes[0].set_xlabel("step"); axes[0].grid(alpha=0.3)
    if dpo_log:
        margins = [h.get("rewards/margins") for h in dpo_log]
        steps = [h["step"] for h in dpo_log]
        axes[1].plot(steps, [m or 0 for m in margins], marker=".", color="C1")
        axes[1].set_title("DPO reward margin (chosen − rejected)")
        axes[1].set_xlabel("step"); axes[1].grid(alpha=0.3)
        axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)


def push_to_hub(local_dir, repo_id, token, base_model):
    from huggingface_hub import HfApi, create_repo
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    # Write README card
    readme = f"""---
base_model: {base_model}
tags: [openenv, dpo, sft, anti-sycophancy, meta-env]
---

# Meta-trained {base_model}

SFT + DPO 2-stage training on the Meta env (long-horizon anti-sycophancy RL environment).

- Base: {base_model}
- Stage 1: SFT on rule-based memory_aware traces
- Stage 2: DPO on synthesized preference pairs (chosen=principled, rejected=sycophant)
- Repo: github.com/SAISriram19/meta

See `eval_results.json` and `training_curves.png` for evidence.
"""
    (local_dir / "README.md").write_text(readme)
    api = HfApi(token=token)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")


if __name__ == "__main__":
    main()
