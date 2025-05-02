#!/usr/bin/env python3
# ppo_llama_trl_pm.py
"""
Fine‑tune a LLaMA‑family model with TRL‑PPO (trl ≤ 0.11)
+ PromptManager.  Reward = Gaussian distance to ground‑truth
+ leading‑digit bonus.
"""

from __future__ import annotations
import argparse, json, math, random, warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers.utils import logging as hf_logging
from tqdm.auto import tqdm
import wandb
from prompt_manager import PromptManager
from torch.nn.utils.rnn import pad_sequence

hf_logging.set_verbosity_error()  # silence HF info


# ------------------------------------------------------------------- #
# reward helper                                                       #
# ------------------------------------------------------------------- #
def mae_digit_reward(
        pred: Optional[float],
        truth: float,
        *,
        digit_weight: float = 0.3,
        max_err: float = 100.0,  # penalty when parse fails
) -> float:
    """
    R = −|pred − truth|                          # MAE term  (higher is better)
        + digit_weight · prefix_match · decay   # partial-credit term

    prefix_match = (# identical leading chars) / len(truth_str)
    decay        = max(0, 1 − |Δ| / (0.6·|truth| + 3))   # fades as error grows
    """
    # 1) handle parse failure
    if pred is None or math.isnan(pred):
        return -max_err

    diff = abs(pred - truth)
    mae_part = -diff  # higher when closer (≤ 0)

    # 2) digit-accuracy bonus
    ps = f"{pred:.4f}".rstrip("0").rstrip(".")
    ts = f"{truth:.4f}".rstrip("0").rstrip(".")
    matched = sum(p == t for p, t in zip(ps, ts))
    prefix_match = matched / len(ts)
    decay = max(0.0, 1.0 - diff / (0.6 * abs(truth) + 3.0))
    digit_bonus = digit_weight * prefix_match * decay

    return mae_part + digit_bonus


def gaussian_reward_digit(
        pred: Optional[float],
        truth: float,
        *,
        sigma: float = 8.0,
        near_bonus: float = 1.0,
        digit_weight: float = 0.1,
) -> float:
    if pred is None or math.isnan(pred):
        return -5.0
    diff = abs(pred - truth)
    base = math.exp(-(diff ** 2) / (2 * sigma * sigma))
    # if diff <= 2.0:
    #    base += near_bonus
    # if diff <= 1.0:
    #    base += near_bonus
    if diff > 25:
        base -= 2.0
    ps, ts = f"{pred:.4f}".rstrip("0").rstrip("."), f"{truth:.4f}".rstrip("0").rstrip(".")
    match = sum(p == t for p, t in zip(ps, ts))
    decay = max(0.0, 1.0 - diff / (3 * sigma))
    digit_acc = match / len(ts)
    return base + digit_weight * digit_acc * decay


# ------------------------------------------------------------------- #
# helpers                                                             #
# ------------------------------------------------------------------- #
def freeze_all_but_last_n(model, n_last: int = 2) -> None:
    if not hasattr(model.pretrained_model, "model") or not hasattr(
            model.pretrained_model.model, "layers"
    ):
        warnings.warn("Could not find transformer layers – skipping freezing.")
        return
    total = len(model.pretrained_model.model.layers)
    for n, p in model.pretrained_model.named_parameters():
        if "layers." in n:
            idx = int(n.split(".")[2])
            p.requires_grad_(idx >= total - n_last)
        else:
            p.requires_grad_(False)
    for n, p in model.named_parameters():
        if "v_head" in n:
            p.requires_grad_(True)


def load_train_dataset(path: str) -> Dataset:
    rows = []
    for ex in json.load(open(path)):
        rows.append({"query": ex["question"], "label": float(ex["answer_value"])})
    return Dataset.from_list(rows)


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


# ------------------------------------------------------------------- #
#  main                                                               #
# ------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/train_data.json")
    parser.add_argument("--model_name_or_path", default="/data/nmysore/out/fine_tuned_model_8r")
    parser.add_argument("--output_dir", default="/data/nmysore/ppo_8r_vo_v1")
    parser.add_argument("--target_steps", type=int, default=8_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--start_temp", type=float, default=2.0)
    parser.add_argument("--end_temp", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_with", choices=["wandb", "tensorboard"], default="wandb")
    args = parser.parse_args()

    # Check and create output directory early to detect permission issues
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory created/verified: {args.output_dir}")
    except PermissionError:
        print(f"ERROR: Permission denied when creating output directory: {args.output_dir}")
        print("Please check directory permissions and try again.")
        return  # Exit early if we can't create the directory

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # tokenizer + prompt manager
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    pm = PromptManager(tok)

    # models
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None,
    )
    policy.pretrained_model.config.dropout = 0.05
    policy.pretrained_model.config.attention_dropout = 0.05
    policy.train()
    ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path, torch_dtype=next(policy.parameters()).dtype
    )
    freeze_all_but_last_n(policy, 4)

    cfg = PPOConfig(
        exp_name="ppo_numeric_reasoning",
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        init_kl_coef=args.kl_coef,
        log_with=args.log_with,
        remove_unused_columns=False,
        whiten_rewards=True,
        target=6,
        horizon=10000,
    )

    dset = load_train_dataset(args.train_file)

    def add_grad_noise(model, std=1e-5):
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * std)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "query": [b["query"] for b in batch],
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        }

    trainer = PPOTrainer(
        config=cfg,
        model=policy,
        ref_model=ref,
        tokenizer=tok,
        dataset=dset,
        data_collator=collate,
    )
    from torch.optim import AdamW
    # ── replace the default Adam with one that has weight-decay ──
    trainer.optimizer = AdamW(
        (p for p in trainer.model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,  # ← 1 % L2 regularisation
    )

    total_updates = math.ceil(args.target_steps / args.batch_size)
    processed, update_idx = 0, 0

    while processed < args.target_steps:
        data_iter = trainer.dataloader
        if trainer.accelerator.is_main_process:
            data_iter = tqdm(data_iter, desc="PPO-epoch", leave=False)

        for batch in data_iter:
            # 1 / temperature schedule
            t_frac   = min(update_idx / max(total_updates - 1, 1), 1.0)
            cur_temp = lerp(args.start_temp, args.end_temp, t_frac)

            # 2 / encode prompts
            prompts = [pm.build_inference_prompt(q) for q in batch["query"]]
            tensors = [torch.tensor(tok.encode(p),
                                    device=trainer.current_device)
                       for p in prompts]

            # 3 / generate two candidate completions
            n_candidates = 2
            with torch.no_grad():
                all_outs = [
                    trainer.generate(
                        tensors,
                        batch_size=len(tensors),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=cur_temp,
                        top_p=0.9,
                        return_prompt=False,
                        pad_to_multiple_of=8,
                    )
                    for _ in range(n_candidates)
                ]

            # 4 / pick best candidate per example + compute reward
            best_ids, rewards = [], []
            for cand_ids, truth in zip(zip(*all_outs), batch["label"]):
                decoded = [tok.decode(ids, skip_special_tokens=True)
                           for ids in cand_ids]

                # per-candidate MAE
                maes = []
                for txt in decoded:
                    guess = pm.extract_carbs_from_answer(
                                pm.parse_cot_and_answer(txt)[1])
                    maes.append(
                        abs(guess - truth.item()) if guess is not None else 1e9
                    )

                k = int(np.argmin(maes))          # index of closer candidate
                best_ids.append(cand_ids[k])

                best_guess = pm.extract_carbs_from_answer(
                                pm.parse_cot_and_answer(decoded[k])[1])
                reward_val = gaussian_reward_digit(best_guess, truth.item())
                rewards.append(torch.tensor(reward_val, dtype=torch.float32))

            outs    = best_ids                     # list[Tensor]
            decoded = [tok.decode(ids, skip_special_tokens=True)
                       for ids in outs]

            # 5 / MAE for logging only
            maes = []
            for text, truth in zip(decoded, batch["label"]):
                guess = pm.extract_carbs_from_answer(
                            pm.parse_cot_and_answer(text)[1])
                if guess is not None and not math.isnan(guess):
                    maes.append(abs(guess - truth.item()))

            # 6 / PPO update  +  tiny grad noise
            stats = trainer.step(tensors, outs, rewards)
            add_grad_noise(policy, 1e-5)

            # 7 / log stats
            trainer.log_stats(
                stats,
                {**batch, "response": decoded},
                rewards,
                columns_to_log=("query", "response"),
            )

            if trainer.accelerator.is_main_process and args.log_with == "wandb":
                mean_mae = float(np.mean(maes)) if maes else np.nan

                rows = []
                for i, (q, r, rew, lbl) in enumerate(
                        zip(batch["query"], decoded, rewards, batch["label"])):
                    uid = processed + i
                    rows.append((
                        uid, q, r,
                        round(float(rew.item()), 3),
                        float(lbl),
                        maes[i] if i < len(maes) else np.nan,
                    ))

                wandb.log(
                    {
                        "metrics/mae": mean_mae,
                        "decoding/temperature": cur_temp,
                        "samples": wandb.Table(
                            columns=["uid", "query", "response",
                                     "reward", "truth", "mae"],
                            rows=rows[:5],
                        ),
                    },
                    step=processed,
                )

            # 8 / book-keeping
            processed  += args.batch_size
            update_idx += 1
            if processed >= args.target_steps:
                break

        if trainer.accelerator.is_main_process:
            print(f"=== processed {processed}/{args.target_steps} ===")

    # save final model
    if trainer.accelerator.is_main_process:
        trainer.save_pretrained(args.output_dir)
        print("✓ model saved to", args.output_dir)


if __name__ == "__main__":
    main()
