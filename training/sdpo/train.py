import os
import copy
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from data.livecodebench import format_question as get_question
from training.common import ValidatorRunConfig
from training.utils import get_world_size, get_grad_norm, get_completion_token_logprobs, build_student_messages, build_teacher_prompt
from validators import Validator

logger = logging.getLogger(__name__)


@dataclass
class SDPOHparams:
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    num_epochs: int = 1
    minibatch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_rollouts: int = 4

    max_prompt_length: int = 2048
    max_response_length: int = 8192

    temperature: float = 1.0
    beta: float = 0.1

    log_interval: int = 10
    validation_interval: int = 10


def compute_dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    reference_chosen_logprobs: torch.Tensor,
    reference_rejected_logprobs: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    policy_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs

    logits = beta * (policy_logratios - reference_logratios)
    losses = -F.logsigmoid(logits)
    loss = losses.mean()

    chosen_rewards = beta * (policy_chosen_logprobs - reference_chosen_logprobs)
    rejected_rewards = beta * (policy_rejected_logprobs - reference_rejected_logprobs)

    metrics = {
        "dpo_margin": (policy_logratios - reference_logratios).mean().item(),
        "preference_accuracy": (logits > 0).float().mean().item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
    }
    return loss, metrics


def teacher_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Dict[str, Any],
    rollouts: List[Dict[str, Any]],
    feedbacks: List[Any],
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gc_was_enabled = model.is_gradient_checkpointing
    if gc_was_enabled:
        model.gradient_checkpointing_disable()

    question = get_question(example)
    prompts = [
        build_teacher_prompt(question, rollout["completion"], feedback.feedback_text)
        for rollout, feedback in zip(rollouts, feedbacks)
    ]
    inputs = tokenizer.apply_chat_template(
        prompts,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_tensors="pt",
        return_in_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.95

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    completions = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results = [
        {"prompt": prompt, "completion": completion}
        for prompt, completion in zip(prompts, completions)
    ]

    tokenizer.padding_side = original_padding_side

    if gc_was_enabled:
        model.gradient_checkpointing_enable()

    return results


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    hparams: SDPOHparams,
    collate_fn: Callable,
    rollout_fn: Callable,
    get_feedback_fn: Callable,
    validators: List[Tuple[Validator, ValidatorRunConfig]],
    output_dir: str = "outputs/SDPO",
    wandb_project: str = "kdpo-exploration",
    wandb_run_name: str = "SDPO",
) -> Dict[str, Any]:
    output_dir = f"{output_dir}/{wandb_run_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    world_size = get_world_size()
    adjusted_gradient_accumulation_steps = hparams.gradient_accumulation_steps // world_size

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
                "beta": hparams.beta,
            },
            init_kwargs={
                "wandb": {"name": wandb_run_name}
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

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    reference_model.to(accelerator.device)
    model.train()

    global_step = 0
    validation_history = defaultdict(list)

    if accelerator.is_main_process:
        logger.info("Beginning training...")
        logger.info(f"World size: {world_size}")
        logger.info(f"Learning rate: {hparams.learning_rate}")
        logger.info(f"Number of training steps: {len(dataloader)}")
        logger.info(f"Number of validation steps: {len(validators)}")
        logger.info(f"Number of epochs: {hparams.num_epochs}")
        logger.info(f"Gradient accumulation steps: {adjusted_gradient_accumulation_steps}")
        logger.info(f"Number of rollouts: {hparams.num_rollouts}")
        logger.info(f"Max prompt length: {hparams.max_prompt_length}")
        logger.info(f"Max response length: {hparams.max_response_length}")
        logger.info(f"Temperature: {hparams.temperature}")
        logger.info(f"Beta: {hparams.beta}")

    for epoch in range(hparams.num_epochs):
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1}/{hparams.num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Processing batch {batch_idx + 1} of {len(dataloader)}...")

            with accelerator.accumulate(model):
                batch_data = []
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
                    feedbacks = [
                        get_feedback_fn(rollout["completion"], example)
                        for rollout in rollouts
                    ]

                    teacher_rollouts = teacher_rollout(
                        unwrapped_model,
                        tokenizer,
                        example,
                        rollouts,
                        feedbacks,
                        temperature=hparams.temperature,
                        max_new_tokens=hparams.max_response_length,
                    )

                    batch_data.extend([
                        {
                            "question": get_question(example),
                            "chosen_completion": teacher_regen["completion"],
                            "rejected_completion": rollout["completion"],
                        }
                        for rollout, teacher_regen in zip(rollouts, teacher_rollouts)
                    ])

                if len(batch_data) == 0:
                    continue

                rejected_messages = [
                    build_student_messages(
                        data["question"], data["rejected_completion"])
                    for data in batch_data
                ]
                chosen_messages = [
                    build_student_messages(
                        data["question"], data["chosen_completion"])
                    for data in batch_data
                ]
                messages = rejected_messages + chosen_messages
                max_seq_length = hparams.max_prompt_length + hparams.max_response_length

                model.train()
                policy_token_logprobs, policy_mask = get_completion_token_logprobs(
                    model,
                    tokenizer,
                    messages,
                    max_seq_length=max_seq_length,
                    requires_grad=True,
                )
                reference_token_logprobs, reference_mask = get_completion_token_logprobs(
                    reference_model,
                    tokenizer,
                    messages,
                    max_seq_length=max_seq_length,
                    requires_grad=False,
                )

                policy_mask = policy_mask.to(policy_token_logprobs.dtype)
                reference_mask = reference_mask.to(reference_token_logprobs.dtype)
                policy_sequence_logprobs = (
                    policy_token_logprobs * policy_mask).sum(dim=-1)
                reference_sequence_logprobs = (
                    reference_token_logprobs * reference_mask).sum(dim=-1)

                pair_count = len(batch_data)
                policy_rejected = policy_sequence_logprobs[:pair_count]
                policy_chosen = policy_sequence_logprobs[pair_count:]
                reference_rejected = reference_sequence_logprobs[:pair_count]
                reference_chosen = reference_sequence_logprobs[pair_count:]

                loss, metrics = compute_dpo_loss(
                    policy_chosen_logprobs=policy_chosen,
                    policy_rejected_logprobs=policy_rejected,
                    reference_chosen_logprobs=reference_chosen,
                    reference_rejected_logprobs=reference_rejected,
                    beta=hparams.beta,
                )

                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients:
                    if hparams.max_grad_norm > 0:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), hparams.max_grad_norm)
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm = grad_norm.item()
                    else:
                        grad_norm = get_grad_norm(model)

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % hparams.log_interval == 0 and accelerator.is_main_process:
                        logger.info(f"Step {global_step} - Loss: {loss.item():.4f}")
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/global_step": global_step,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/completion_tokens": policy_mask.sum().item(),
                            "train/dpo_margin": metrics["dpo_margin"],
                            "train/preference_accuracy": metrics["preference_accuracy"],
                            "train/chosen_reward": metrics["chosen_reward"],
                            "train/rejected_reward": metrics["rejected_reward"],
                            "train/reward_margin": metrics["reward_margin"],
                        }
                        if grad_norm is not None:
                            log_dict["train/grad_norm"] = grad_norm
                        accelerator.log(log_dict, step=global_step)

                    if global_step % hparams.validation_interval == 0 and accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.eval()
                        original_padding_side = tokenizer.padding_side
                        original_pad_token = tokenizer.pad_token
                        original_pad_token_id = tokenizer.pad_token_id

                        try:
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
                        finally:
                            tokenizer.padding_side = original_padding_side
                            tokenizer.pad_token = original_pad_token
                            tokenizer.pad_token_id = original_pad_token_id

                        model.train()

                    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Running final validation...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()
        original_padding_side = tokenizer.padding_side
        original_pad_token = tokenizer.pad_token
        original_pad_token_id = tokenizer.pad_token_id

        try:
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
                        {"step": "final", "score": score}
                    )
                    logger.info(
                        f"Final validation score for {validator.name}: {score:.4f}")
                except Exception as e:
                    logger.error(f"Error validating with {validator.name}: {e}")
        finally:
            tokenizer.padding_side = original_padding_side
            tokenizer.pad_token = original_pad_token
            tokenizer.pad_token_id = original_pad_token_id

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
    from data.livecodebench import LiveCodeBenchDataset, collate_fn as lcb_collate_fn, rollout as lcb_rollout, get_environment_feedback as lcb_get_feedback
    from validators import LiveCodeBenchValidator, FineWebValidator

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/SDPO")
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

    hparams = SDPOHparams(
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
        beta=args.beta,
        log_interval=args.log_interval,
        validation_interval=args.validation_interval,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.gradient_checkpointing_enable()

    dataset = LiveCodeBenchDataset()
    collate_fn = lcb_collate_fn
    rollout_fn = lcb_rollout
    get_feedback_fn = lcb_get_feedback

    validators = [
        (LiveCodeBenchValidator(), ValidatorRunConfig(
            batch_size=8,
            max_new_tokens=2048,
            max_seq_length=2048
        )),
        (FineWebValidator(), ValidatorRunConfig(
            batch_size=16,
            max_new_tokens=0,
            max_seq_length=1024
        )),
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
