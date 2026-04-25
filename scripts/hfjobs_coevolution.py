"""HF Jobs entrypoint — 3-round CoEvolution training loop.

Round 0: rule-based eval (memory_aware vs template adversary) → baseline
Round 1: train AGENT on round-0 failures (DPO) → eval → capture new failures
Round 2: train ADVERSARY on round-1 successes (where attacks didn't land) → eval
Round 3: train AGENT again on round-2 harder pairs → final eval

Uploads each round's checkpoints + cross-round comparison chart to HF Hub.

Reference: CoEvolve (arXiv 2604.15840) — agent-data mutual evolution. Their
result: +19.4% on AppWorld via 3-round loop. We implement the loop in our env.

Usage from `hf jobs uv run`:
    hf jobs uv run \\
      --flavor a10g-large \\  # need more VRAM for two LoRAs
      --with transformers --with peft --with trl --with bitsandbytes \\
      --with datasets --with accelerate --with matplotlib --with pyyaml \\
      --with networkx --with pydantic --with huggingface_hub \\
      -s HF_TOKEN \\
      https://raw.githubusercontent.com/SAISriram19/meta/main/scripts/hfjobs_coevolution.py

Env vars:
    HF_TOKEN              required
    HUB_REPO_ID           required, e.g. "username/meta-coevolution"
    AGENT_MODEL           default Qwen/Qwen2.5-3B-Instruct
    ADVERSARY_MODEL       default Qwen/Qwen2.5-3B-Instruct
    SCENARIOS             default L2_strategic_shift,L4_market_pivot
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/SAISriram19/meta.git"
WORK = Path("/tmp/meta")


def sh(cmd):
    print(f"[sh] {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True)


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
    if not (HUB_REPO_ID and HF_TOKEN):
        raise SystemExit("HUB_REPO_ID and HF_TOKEN required")
    AGENT_MODEL = os.environ.get("AGENT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    ADVERSARY_MODEL = os.environ.get("ADVERSARY_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    SCENARIOS = os.environ.get("SCENARIOS", "L2_strategic_shift,L4_market_pivot").split(",")

    print(f"[config] agent={AGENT_MODEL} adv={ADVERSARY_MODEL} scenarios={SCENARIOS}")

    clone_repo()
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    # --- Build initial DPO data + adversary preference pairs ---
    sh([sys.executable, "scripts/build_dpo_pairs.py", "--out", "data/agent_pairs.jsonl",
        "--cap-per-scenario", "100"])
    sh([sys.executable, "scripts/build_adversary_dpo_pairs.py",
        "--out", "data/adv_pairs.jsonl"])

    # --- Round 0: baseline eval ---
    print("\n=== ROUND 0: BASELINE ===")
    r0_results = run_baseline_eval(SCENARIOS)
    save_results("/tmp/r0_baseline.json", r0_results)

    # --- SFT warm-start (cold-start DPO failed for us; SFT first matches
    #     DeepSeek-R1 + Tülu 3 recipe). Mines memory_aware traces on L0+L2+L4. ---
    print("\n=== SFT WARM-START ===")
    sft_data_path = "/tmp/sft_traces.jsonl"
    n_sft = build_sft_traces(SCENARIOS, sft_data_path, seeds=(0, 1, 2))
    print(f"[sft-data] {n_sft} traces from rule-based memory_aware")
    sft_lora = sft_train(
        model=AGENT_MODEL,
        sft_data_path=sft_data_path,
        output="/tmp/sft_warmstart_lora",
        epochs=1,
    )

    # --- Round 1: train agent (DPO on top of SFT'd weights) ---
    print("\n=== ROUND 1: TRAIN AGENT (DPO from SFT warm-start) ===")
    agent_lora = train_dpo(
        model=AGENT_MODEL,
        data="data/agent_pairs.jsonl",
        output="/tmp/r1_agent_lora",
        epochs=2,
        lr=5e-6,
        from_lora=sft_lora,  # continue from SFT'd weights, not raw base
    )
    r1_results = eval_with_lora(agent_lora, AGENT_MODEL, SCENARIOS, "r1_trained_agent")
    save_results("/tmp/r1_agent_eval.json", r1_results)

    # --- Round 2: train adversary on what's still working for the agent ---
    print("\n=== ROUND 2: TRAIN ADVERSARY ===")
    adv_lora = train_dpo(
        model=ADVERSARY_MODEL,
        data="data/adv_pairs.jsonl",
        output="/tmp/r2_adversary_lora",
        epochs=2,
        lr=5e-6,
    )
    # We could re-eval agent against trained adversary here, but the env
    # would need to load the LoRA — defer to round 3 final eval.

    # --- Mine round-1 failures: where did the trained agent still cave? ---
    print("\n=== MINING ROUND-1 FAILURES ===")
    new_pairs_path = "/tmp/r1_failures.jsonl"
    n_mined = mine_failures(agent_lora, AGENT_MODEL, SCENARIOS, new_pairs_path,
                             adversary_lora=None)  # template adversary in round 1
    print(f"[mined] {n_mined} new failure pairs")

    # --- Round 3: retrain agent on round-1 failures + use TRAINED ADVERSARY ---
    print("\n=== ROUND 3: RETRAIN AGENT vs TRAINED ADVERSARY ===")
    # Combine round-1 pairs with new failure pairs for richer training.
    combined_pairs_path = "/tmp/r3_combined_pairs.jsonl"
    combine_pair_files(["data/agent_pairs.jsonl", new_pairs_path], combined_pairs_path)

    agent_lora_v2 = train_dpo(
        model=AGENT_MODEL,
        data=combined_pairs_path,
        output="/tmp/r3_agent_lora",
        epochs=3,
        lr=4e-6,
    )
    # Eval round-3 agent AGAINST trained adversary (LoRA-driven)
    r3_results = eval_with_lora(
        agent_lora_v2, AGENT_MODEL, SCENARIOS, "r3_trained_agent",
        adversary_lora=adv_lora,
    )
    save_results("/tmp/r3_agent_eval.json", r3_results)

    # --- Plot cross-round comparison ---
    plot_coevolution_curve(r0_results, r1_results, r3_results, "/tmp/coevolution_curve.png")

    # --- Push everything to Hub ---
    push_to_hub(HUB_REPO_ID, HF_TOKEN, AGENT_MODEL,
                {"r1_agent_lora": agent_lora, "r2_adversary_lora": adv_lora,
                 "r3_agent_lora": agent_lora_v2},
                {"r0_baseline.json": "/tmp/r0_baseline.json",
                 "r1_agent_eval.json": "/tmp/r1_agent_eval.json",
                 "r3_agent_eval.json": "/tmp/r3_agent_eval.json",
                 "coevolution_curve.png": "/tmp/coevolution_curve.png"})


def run_baseline_eval(scenarios):
    """Eval rule-based baselines + capture per-competency."""
    from env.environment import StakeholderEnv
    from eval.harness import rollout
    from eval.policies import build_policy
    from eval.competencies import score_rollout

    rows = []
    for sc in scenarios:
        for pname in ["sycophant", "memory_aware"]:
            for seed in (0, 1):
                env = StakeholderEnv()
                rec = rollout(env, pname, build_policy(pname), sc, seed=seed, capture_trace=True)
                comp = score_rollout(rec._trace, sc)
                rows.append({
                    "round": 0,
                    "policy": pname,
                    "scenario": sc,
                    "seed": seed,
                    "reward": rec.total_reward,
                    "bad_agreements": rec.bad_agreements,
                    "principled": rec.principled_pushbacks,
                    "AR": comp["AR"]["score"],
                    "TTL": comp["TTL"]["score"],
                    "LRU": comp["LRU"]["score"],
                    "CR": comp["CR"]["score"],
                    "composite": comp["composite"],
                })
    return rows


def build_sft_traces(scenarios, out_path, seeds=(0, 1, 2)):
    """Roll rule-based memory_aware on each (scenario, seed); save (prompt, completion) pairs.

    Used for SFT warm-start so round-1 DPO doesn't cold-start on Qwen 3B base.
    """
    from env.environment import StakeholderEnv
    from eval.policies import build_policy
    from eval.harness import RolloutContext
    from scripts.train import SYSTEM_PROMPT, format_prompt

    policy = build_policy("memory_aware")
    rows = []
    for sc in scenarios:
        for seed in seeds:
            env = StakeholderEnv()
            obs = env.reset(task_id=sc)
            step = 0
            while not env.done and step < env.scenario.step_budget:
                step += 1
                ctx = RolloutContext(observation=obs, env=env, step_no=step)
                action = policy(ctx)
                prompt = SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + format_prompt(obs, env)
                action_json = json.dumps(action.model_dump())
                rows.append({"prompt": prompt, "completion": action_json})
                r = env.step(action)
                obs = r.observation
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def sft_train(model, sft_data_path, output, epochs):
    """SFT cold-start training. Returns LoRA path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    lc = LoraConfig(r=32, lora_alpha=64,
                    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model_peft = get_peft_model(base, lc)

    rows = [json.loads(l) for l in open(sft_data_path)]
    texts = [r["prompt"] + "\n\nACTION:\n" + r["completion"] + tok.eos_token for r in rows]
    ds = Dataset.from_list([{"text": t} for t in texts])

    base_kwargs = dict(
        output_dir=output, num_train_epochs=epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=1e-5, logging_steps=2, save_steps=200, seed=42, bf16=True,
        report_to="none", max_seq_length=2048, dataset_text_field="text",
    )
    while True:
        try:
            cfg = SFTConfig(**base_kwargs); break
        except TypeError as e:
            import re
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m: raise
            base_kwargs.pop(m.group(1), None)
    tk = dict(model=model_peft, train_dataset=ds, args=cfg)
    try:
        trainer = SFTTrainer(processing_class=tok, **tk)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tok, **tk)
    trainer.train()
    out_dir = f"{output}-lora"
    model_peft.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    # Free GPU before next stage
    del model_peft, base, trainer
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return out_dir


def train_dpo(model, data, output, epochs, lr, from_lora=None):
    """Wraps train_dpo_ddp.py logic in-process. Returns LoRA path.

    If `from_lora` is given, loads that LoRA on top of the base before adding
    a NEW LoRA layer for DPO (cold-start cure).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    if from_lora:
        # Apply the prior LoRA, merge into base, then add a fresh LoRA on top.
        prior = PeftModel.from_pretrained(base, from_lora)
        try:
            base = prior.merge_and_unload()
        except Exception:
            base = prior  # if merge fails, training continues on stacked LoRAs
    lc = LoraConfig(r=32, lora_alpha=64,
                    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model_peft = get_peft_model(base, lc)

    rows = [json.loads(l) for l in open(data)]
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])

    base_kwargs = dict(
        output_dir=output, num_train_epochs=epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=lr, beta=0.1, max_prompt_length=1500, max_length=1700,
        logging_steps=1, save_steps=100, seed=42, bf16=True,
        report_to="none", remove_unused_columns=False,
    )
    while True:
        try:
            cfg = DPOConfig(**base_kwargs); break
        except TypeError as e:
            import re
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m: raise
            base_kwargs.pop(m.group(1), None)

    tk = dict(model=model_peft, train_dataset=ds, args=cfg)
    try:
        trainer = DPOTrainer(processing_class=tok, **tk)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tok, **tk)
    trainer.train()
    out_dir = f"{output}-lora"
    model_peft.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def eval_with_lora(lora_path, base_model, scenarios, policy_name, adversary_lora=None):
    """Load LoRA + eval on scenarios with trace capture.

    If `adversary_lora` is provided, plug a trained LLM-adversary driver into
    the env (real two-player co-evolution at eval time).
    """
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
    tok = AutoTokenizer.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()

    def policy(ctx):
        prompt = LLM_SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + format_prompt(ctx.observation, ctx.env) + "\n\nReturn ONE action as strict JSON."
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1800).to(model.device)
        out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                              temperature=0.4, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_completion(text, ctx.env)

    rows = []
    for sc in scenarios:
        for seed in (0, 1):
            env = StakeholderEnv()
            env.reset(task_id=sc)
            # Optionally swap in trained-LLM adversary AFTER reset.
            if adversary_lora and env.scenario.adversarial_stakeholder_id:
                from env.trainable_adversary import LLMAdversaryDriver
                spec = next((s for s in env.scenario.stakeholders
                             if s.id == env.scenario.adversarial_stakeholder_id), None)
                if spec:
                    drv = LLMAdversaryDriver(spec=spec, model_path=adversary_lora)
                    env.set_adversary_driver(drv)
            rec = rollout(env, policy_name, policy, sc, seed=seed, capture_trace=True)
            comp = score_rollout(rec._trace, sc)
            rows.append({
                "round": int(policy_name[1]) if policy_name[0] == "r" else 0,
                "policy": policy_name,
                "scenario": sc,
                "seed": seed,
                "reward": rec.total_reward,
                "bad_agreements": rec.bad_agreements,
                "principled": rec.principled_pushbacks,
                "AR": comp["AR"]["score"],
                "TTL": comp["TTL"]["score"],
                "LRU": comp["LRU"]["score"],
                "CR": comp["CR"]["score"],
                "composite": comp["composite"],
                "adversary": "trained_lora" if adversary_lora else "template",
            })
    # Free GPU memory before next stage
    del model, base
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return rows


def mine_failures(agent_lora, base_model, scenarios, out_path, adversary_lora=None):
    """Run the trained agent and mine cases where it failed. Build new
    (prompt, chosen, rejected) pairs from those failures.

    chosen   = principled action that should have been taken
    rejected = the (cave-style) sycophant action

    Result: new DPO pairs targeted at the agent's CURRENT weak spots.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from env.environment import StakeholderEnv
    from env.models import GroundTruthTag
    from eval.policies import LLM_SYSTEM_PROMPT
    from scripts.train import SYSTEM_PROMPT, format_prompt, parse_completion
    from scripts.build_dpo_pairs import (
        principled_action_for, sycophant_action_for, overrefusal_action_for,
    )

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(agent_lora)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, agent_lora)
    model.eval()

    new_pairs = []
    for sc in scenarios:
        env = StakeholderEnv()
        obs = env.reset(task_id=sc)
        # Optional adversary swap
        if adversary_lora and env.scenario.adversarial_stakeholder_id:
            from env.trainable_adversary import LLMAdversaryDriver
            spec = next((s for s in env.scenario.stakeholders
                         if s.id == env.scenario.adversarial_stakeholder_id), None)
            if spec:
                env.set_adversary_driver(LLMAdversaryDriver(spec=spec, model_path=adversary_lora))

        step = 0
        while not env.done and step < env.scenario.step_budget:
            step += 1
            prompt_text = LLM_SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + format_prompt(obs, env) + "\n\nReturn ONE action as strict JSON."
            inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=1800).to(model.device)
            out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                                  temperature=0.4, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            action = parse_completion(text, env)
            r = env.step(action)
            # Detect a FAILURE: agent took an action with reward <= -0.1 OR
            # the inbound was BAD and the agent's stance was "agree"
            if obs.new_messages:
                addressed = obs.new_messages[-1]
                tag = addressed.ground_truth_tag
                from env.models import SendMessageAction, Stance
                cave = (isinstance(action, SendMessageAction) and
                         action.stance == Stance.AGREE and
                         tag == GroundTruthTag.BAD)
                if cave or r.reward < -0.1:
                    # Build new pair
                    sft_prompt = SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + format_prompt(obs, env)
                    chosen = principled_action_for(addressed, abs(hash(sft_prompt)))
                    rejected = sycophant_action_for(addressed)
                    new_pairs.append({
                        "prompt": sft_prompt,
                        "chosen": json.dumps(chosen),
                        "rejected": json.dumps(rejected),
                        "scenario": sc,
                        "tag": "mined_failure",
                        "kind": "send_message",
                    })
            obs = r.observation

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in new_pairs:
            f.write(json.dumps(p) + "\n")

    # Free GPU
    del model, base
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return len(new_pairs)


def combine_pair_files(input_paths, out_path):
    """Concatenate JSONL pair files, dedup by (prompt, chosen, rejected)."""
    seen = set()
    rows = []
    for p in input_paths:
        if not Path(p).exists():
            continue
        for line in open(p):
            d = json.loads(line)
            key = (d.get("prompt", ""), d.get("chosen", ""), d.get("rejected", ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(d)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[combined] {len(rows)} unique pairs in {out_path}")


def save_results(path, rows):
    Path(path).write_text(json.dumps(rows, indent=2))
    print(f"[saved] {path}")


def plot_coevolution_curve(r0, r1, r3, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [0, 1, 3]
    # Mean reward per round (memory_aware in r0, trained in r1/r3)
    def mean_reward(rows, policy_filter=None):
        if policy_filter:
            rows = [r for r in rows if policy_filter in r["policy"]]
        return sum(r["reward"] for r in rows) / max(1, len(rows))

    def mean_competency(rows, key):
        return sum(r[key] for r in rows) / max(1, len(rows))

    rewards = [
        mean_reward(r0, "memory_aware"),
        mean_reward(r1),
        mean_reward(r3),
    ]
    composite = [mean_competency(r0, "composite"), mean_competency(r1, "composite"), mean_competency(r3, "composite")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(rounds, rewards, marker="o", linewidth=2, color="C0")
    axes[0].set_title("Mean reward across coevolution rounds")
    axes[0].set_xlabel("round"); axes[0].set_ylabel("reward")
    axes[0].grid(alpha=0.3); axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

    axes[1].plot(rounds, composite, marker="s", linewidth=2, color="C1")
    axes[1].set_title("Composite competency score (AR/TTL/LRU/CR avg)")
    axes[1].set_xlabel("round"); axes[1].set_ylabel("composite [0, 1]")
    axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[saved] {out_path}")


def push_to_hub(repo_id, token, base_model, lora_dirs, files):
    from huggingface_hub import HfApi, create_repo
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    api = HfApi(token=token)
    for name, dir_path in lora_dirs.items():
        if Path(dir_path).exists():
            api.upload_folder(
                folder_path=dir_path, repo_id=repo_id,
                repo_type="model", path_in_repo=name,
            )
    for name, src in files.items():
        if Path(src).exists():
            api.upload_file(path_or_fileobj=src, path_in_repo=name,
                            repo_id=repo_id, repo_type="model")
    # README card
    readme = f"""---
base_model: {base_model}
tags: [openenv, dpo, coevolution, anti-sycophancy, meta-env, long-horizon]
---

# Meta env — 3-round CoEvolution training

Two-player co-evolution: agent + adversary trained iteratively against each other.

- Round 0: rule-based baseline
- Round 1: agent trained via DPO on initial preference pairs
- Round 2: adversary trained via DPO on successful attacks
- Round 3: agent retrained with refined hyperparameters

Per-competency scores follow MemoryAgentBench (ICLR 2026, arXiv 2507.05257).

See `coevolution_curve.png` and `r{{0,1,3}}_*_eval.json` for evidence.
"""
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")
    print(f"[done] pushed everything to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
