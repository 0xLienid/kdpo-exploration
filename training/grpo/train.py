import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.livecodebench import format_question as get_question
from training.train import Hparams, ValidatorRunConfig, train as run_train
from training.utils import build_reference_model, build_student_messages, get_completion_token_logprobs


logger = logging.getLogger(__name__)


@dataclass
class GRPOHparams(Hparams):
    num_rollouts: int = 8
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    temperature: float = 1.0
    clip_epsilon: float = 0.2
    kl_coef: float = 0.01
    advantage_epsilon: float = 1e-6


def compute_grpo_loss(
    new_token_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    ref_token_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    kl_coef: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask = token_mask.to(new_token_logprobs.dtype)
    token_count = mask.sum().clamp(min=1.0)

    ratios = torch.exp(new_token_logprobs - old_token_logprobs)
    expanded_advantages = advantages.unsqueeze(1)
    unclipped = ratios * expanded_advantages
    clipped = torch.clamp(ratios, 1.0 - clip_epsilon,
                          1.0 + clip_epsilon) * expanded_advantages
    policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / token_count

    kl = ((new_token_logprobs - ref_token_logprobs) * mask).sum() / token_count
    loss = policy_loss + (kl_coef * kl)

    clip_fraction = ((ratios > (1.0 + clip_epsilon)) |
                     (ratios < (1.0 - clip_epsilon))).to(mask.dtype)
    clip_fraction = (clip_fraction * mask).sum() / token_count

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl": kl.item(),
        "clip_fraction": clip_fraction.item(),
        "ratio_mean": ((ratios * mask).sum() / token_count).item(),
    }
    return loss, metrics


def forward(
    accelerator,
    model: AutoModelForCausalLM,
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    hparams: GRPOHparams,
    reference_model: AutoModelForCausalLM,
    rollout_fn: Callable,
    get_feedback_fn: Callable,
) -> Optional[Dict[str, Any]]:
    batch_data: List[Dict[str, Any]] = []
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    for example in batch:
        rollouts = rollout_fn(
            unwrapped_model,
            tokenizer,
            example,
            num_rollouts=hparams.num_rollouts,
            temperature=hparams.temperature,
            max_new_tokens=hparams.max_response_length,
        )

        rewards: List[float] = []
        feedbacks = []
        for rollout in rollouts:
            feedback = get_feedback_fn(rollout["completion"], example)
            feedbacks.append(feedback)
            all_passed = bool(feedback.metadata.get(
                "all_passed", feedback.success)) if feedback.metadata else bool(feedback.success)
            rewards.append(1.0 if all_passed else 0.0)

        reward_tensor = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=accelerator.device,
        )
        reward_mean = reward_tensor.mean()
        reward_std = reward_tensor.std(unbiased=False)
        advantages = (reward_tensor - reward_mean) / (
            reward_std + hparams.advantage_epsilon
        )

        for rollout, feedback, reward, advantage in zip(
            rollouts, feedbacks, rewards, advantages.tolist()
        ):
            batch_data.append(
                {
                    "question": get_question(example),
                    "completion": rollout["completion"],
                    "feedback": feedback.feedback_text,
                    "reward": float(reward),
                    "advantage": float(advantage),
                }
            )

    if not batch_data:
        return None

    messages = [
        build_student_messages(data["question"], data["completion"])
        for data in batch_data
    ]
    max_seq_length = hparams.max_prompt_length + hparams.max_response_length

    model.train()
    new_token_logprobs, token_mask = get_completion_token_logprobs(
        model,
        tokenizer,
        messages,
        max_seq_length=max_seq_length,
        requires_grad=True,
    )
    old_token_logprobs = new_token_logprobs.detach()
    ref_token_logprobs, _ = get_completion_token_logprobs(
        reference_model,
        tokenizer,
        messages,
        max_seq_length=max_seq_length,
        requires_grad=False,
    )

    advantages = torch.tensor(
        [data["advantage"] for data in batch_data],
        dtype=new_token_logprobs.dtype,
        device=new_token_logprobs.device,
    )
    rewards = torch.tensor(
        [data["reward"] for data in batch_data],
        dtype=new_token_logprobs.dtype,
        device=new_token_logprobs.device,
    )

    return {
        "new_token_logprobs": new_token_logprobs,
        "old_token_logprobs": old_token_logprobs,
        "ref_token_logprobs": ref_token_logprobs,
        "token_mask": token_mask,
        "advantages": advantages,
        "rewards": rewards,
    }


def make_forward_backward_fn(
    rollout_fn: Callable,
    get_feedback_fn: Callable,
) -> Callable:
    def forward_backward(
        accelerator,
        model: AutoModelForCausalLM,
        batch: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        hparams: GRPOHparams,
        auxiliary_model: AutoModelForCausalLM,
    ):
        forward_outputs = forward(
            accelerator,
            model,
            batch,
            tokenizer,
            hparams,
            auxiliary_model,
            rollout_fn,
            get_feedback_fn,
        )
        if forward_outputs is None:
            return None, {}

        loss, metrics = compute_grpo_loss(
            new_token_logprobs=forward_outputs["new_token_logprobs"],
            old_token_logprobs=forward_outputs["old_token_logprobs"],
            ref_token_logprobs=forward_outputs["ref_token_logprobs"],
            token_mask=forward_outputs["token_mask"],
            advantages=forward_outputs["advantages"],
            clip_epsilon=hparams.clip_epsilon,
            kl_coef=hparams.kl_coef,
        )
        metrics.update(
            {
                "reward_mean": forward_outputs["rewards"].mean().item(),
                "reward_std": forward_outputs["rewards"].std(unbiased=False).item(),
                "advantage_mean": forward_outputs["advantages"].mean().item(),
                "advantage_std": forward_outputs["advantages"].std(unbiased=False).item(),
                "completion_tokens": forward_outputs["token_mask"].sum().item(),
            }
        )
        return loss, metrics

    return forward_backward


if __name__ == "__main__":
    import argparse
    import random

    import numpy as np

    from data.livecodebench import (
        LiveCodeBenchDataset,
        collate_fn as lcb_collate_fn,
        get_environment_feedback as lcb_get_feedback,
        rollout as lcb_rollout,
    )
    from validators import FineWebValidator, LiveCodeBenchValidator

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--advantage-epsilon", type=float, default=1e-6)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/grpo")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    hparams = GRPOHparams(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        minibatch_size=args.minibatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_rollouts=args.num_rollouts,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
        clip_epsilon=args.clip_epsilon,
        kl_coef=args.kl_coef,
        advantage_epsilon=args.advantage_epsilon,
        log_interval=args.log_interval,
        validation_interval=args.validation_interval,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.gradient_checkpointing_enable()

    reference_model = build_reference_model(model)
    forward_backward_fn = make_forward_backward_fn(
        lcb_rollout,
        lcb_get_feedback,
    )

    dataset = LiveCodeBenchDataset()
    validators = [
        (
            LiveCodeBenchValidator(),
            ValidatorRunConfig(
                batch_size=8,
                max_new_tokens=2048,
                max_seq_length=2048,
            ),
        ),
        (
            FineWebValidator(),
            ValidatorRunConfig(
                batch_size=16,
                max_new_tokens=0,
                max_seq_length=1024,
            ),
        ),
    ]

    run_train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        hparams=hparams,
        collate_fn=lcb_collate_fn,
        forward_backward_fn=forward_backward_fn,
        validators=validators,
        auxiliary_model=reference_model,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
