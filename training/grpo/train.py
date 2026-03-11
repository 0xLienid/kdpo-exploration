import os
import copy
import logging
import torch
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, GradientAccumulationPlugin
from training.common import ValidatorRunConfig
from training.utils import get_world_size, build_student_messages, get_completion_token_logprobs
from validators.validator import Validator

logger = logging.getLogger(__name__)


@dataclass
class GRPOHparams:
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    num_epochs: int = 1
    minibatch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_rollouts: int = 8

    max_prompt_length: int = 2048
    max_response_length: int = 8192

    temperature: float = 1.0
    clip_epsilon: float = 0.2
    kl_coef: float = 0.01
    advantage_epsilon: float = 1e-6

    log_interval: int = 10
    validation_interval: int = 10


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
    policy_loss = -(torch.minimum(unclipped, clipped)
                    * mask).sum() / token_count

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


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    hparams: GRPOHparams,
    collate_fn: Callable,
    rollout_fn: Callable,
    get_feedback_fn: Callable,
    validators: List[Tuple[Validator, ValidatorRunConfig]],
    output_dir: str = "outputs/grpo",
    wandb_project: str = "kdpo-exploration",
    wandb_run_name: str = "grpo",
) -> Dict[str, Any]:
    output_dir = f"{output_dir}/{wandb_run_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    world_size = get_world_size()
    adjusted_gradient_accumulation_steps = max(
        1, hparams.gradient_accumulation_steps // world_size)

    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=adjusted_gradient_accumulation_steps,
        adjust_scheduler=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        log_with="wandb" if wandb_project is not None else None,
    )

    if accelerator.is_main_process and wandb_project:
        accelerator.init_trackers(
            project_name=wandb_project,
            config={
                "learning_rate": hparams.learning_rate,
                "num_epochs": hparams.num_epochs,
                "minibatch_size": hparams.minibatch_size,
                "gradient_accumulation_steps": adjusted_gradient_accumulation_steps,
                "num_rollouts": hparams.num_rollouts,
                "max_prompt_length": hparams.max_prompt_length,
                "max_response_length": hparams.max_response_length,
                "temperature": hparams.temperature,
                "clip_epsilon": hparams.clip_epsilon,
                "kl_coef": hparams.kl_coef,
                "advantage_epsilon": hparams.advantage_epsilon,
            },
            init_kwargs={"wandb": {"name": wandb_run_name}
                         } if wandb_run_name else None,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=hparams.minibatch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )

    reference_model = copy.deepcopy(model)
    if hasattr(reference_model, "gradient_checkpointing_disable"):
        reference_model.gradient_checkpointing_disable()
    reference_model.requires_grad_(False)
    reference_model.eval()

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader)
    reference_model.to(accelerator.device)

    model.train()
    global_step = 0
    validation_history = defaultdict(list)

    if accelerator.is_main_process:
        logger.info("Beginning training...")
        logger.info(f"Number of training steps: {len(dataloader)}")
        logger.info(f"Number of validation steps: {len(validators)}")
        logger.info(f"Number of epochs: {hparams.num_epochs}")
        logger.info(
            f"Gradient accumulation steps: {adjusted_gradient_accumulation_steps}")
        logger.info(f"Number of rollouts: {hparams.num_rollouts}")
        logger.info(f"Max prompt length: {hparams.max_prompt_length}")
        logger.info(f"Max response length: {hparams.max_response_length}")
        logger.info(f"Temperature: {hparams.temperature}")
        logger.info(f"Clip epsilon: {hparams.clip_epsilon}")
        logger.info(f"KL coefficient: {hparams.kl_coef}")

    for epoch in range(hparams.num_epochs):
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1}/{hparams.num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            logger.info(
                f"Processing batch {batch_idx + 1} of {len(dataloader)}...")

            with accelerator.accumulate(model):
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
                        feedback = get_feedback_fn(
                            rollout["completion"], example)
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
                    advantages = (reward_tensor - reward_mean) / \
                        (reward_std + hparams.advantage_epsilon)

                    for rollout, feedback, reward, advantage in zip(rollouts, feedbacks, rewards, advantages.tolist()):
                        batch_data.append(
                            {
                                "question": example["question"],
                                "completion": rollout["completion"],
                                "feedback": feedback.feedback_text,
                                "reward": float(reward),
                                "advantage": float(advantage),
                            }
                        )

                if len(batch_data) == 0:
                    continue

                messages = [
                    build_student_messages(
                        data["question"], data["completion"])
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

                loss, metrics = compute_grpo_loss(
                    new_token_logprobs=new_token_logprobs,
                    old_token_logprobs=old_token_logprobs,
                    ref_token_logprobs=ref_token_logprobs,
                    token_mask=token_mask,
                    advantages=advantages,
                    clip_epsilon=hparams.clip_epsilon,
                    kl_coef=hparams.kl_coef,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if hparams.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(
                            model.parameters(), hparams.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1

                    if global_step % hparams.log_interval == 0 and accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step} - Loss: {loss.item():.4f}")
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/policy_loss": metrics["policy_loss"],
                            "train/kl": metrics["kl"],
                            "train/clip_fraction": metrics["clip_fraction"],
                            "train/ratio_mean": metrics["ratio_mean"],
                            "train/reward_mean": rewards.mean().item(),
                            "train/reward_std": rewards.std(unbiased=False).item(),
                            "train/advantage_mean": advantages.mean().item(),
                            "train/advantage_std": advantages.std(unbiased=False).item(),
                            "train/completion_tokens": token_mask.sum().item(),
                            "train/global_step": global_step,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                        }
                        accelerator.log(log_dict, step=global_step)

                    if global_step % hparams.validation_interval == 0 and accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.eval()

                        for validator, val_config in validators:
                            logger.info(f"Validating with {validator.name}...")
                            try:
                                score = validator.validate(
                                    model=unwrapped_model,
                                    tokenizer=tokenizer,
                                    batch_size=val_config.batch_size,
                                    max_new_tokens=val_config.max_new_tokens,
                                    max_seq_length=val_config.max_seq_length,
                                )
                                validation_history[validator.name].append(
                                    {"step": global_step, "score": score}
                                )
                                accelerator.log(
                                    {f"val/{validator.name}": score}, step=global_step)
                                logger.info(
                                    f"Validation score for {validator.name}: {score:.4f}")
                            except Exception as e:
                                logger.error(
                                    f"Error validating with {validator.name}: {e}")

                        model.train()

                    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Running final validation...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()

        for validator, val_config in validators:
            logger.info(f"Validating with {validator.name}...")
            try:
                score = validator.validate(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    batch_size=val_config.batch_size,
                    max_new_tokens=val_config.max_new_tokens,
                    max_seq_length=val_config.max_seq_length,
                )
                validation_history[validator.name].append(
                    {"step": "final", "score": score})
                logger.info(
                    f"Final validation score for {validator.name}: {score:.4f}")
            except Exception as e:
                logger.error(f"Error validating with {validator.name}: {e}")

        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
        logger.info("Training complete!")

        if wandb_project:
            accelerator.end_training()

    accelerator.wait_for_everyone()

    return {
        "validation_history": validation_history,
    }


if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    from data.livecodebench import (
        LiveCodeBenchDataset,
        collate_fn as lcb_collate_fn,
        rollout as lcb_rollout,
        get_environment_feedback as lcb_get_feedback,
    )
    from validators import LiveCodeBenchValidator, FineWebValidator

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=1)
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

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = LiveCodeBenchDataset()
    collate_fn = lcb_collate_fn
    rollout_fn = lcb_rollout
    get_feedback_fn = lcb_get_feedback

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

    train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        hparams=hparams,
        collate_fn=collate_fn,
        rollout_fn=rollout_fn,
        get_feedback_fn=get_feedback_fn,
        validators=validators,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
