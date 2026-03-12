import os
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
from training.utils import get_world_size, get_grad_norm, get_logits_completion_ids_and_mask, build_student_messages, build_teacher_messages
from training.opsd.EMATeacher import EMATeacher
from validators import Validator

logger = logging.getLogger(__name__)


@dataclass
class OPSDHparams:
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    num_epochs: int = 1
    minibatch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_rollouts: int = 8

    max_prompt_length: int = 2048
    max_response_length: int = 8192

    teacher_alpha: float = 0.01
    top_k: int = 20
    temperature: float = 1.0

    log_interval: int = 10
    validation_interval: int = 10


def compute_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    k: int = 20
) -> torch.Tensor:
    student_topk_logits, student_topk_indices = torch.topk(
        student_logits, k, dim=-1)
    teacher_logits_at_topk_indices = torch.gather(
        teacher_logits, dim=-1, index=student_topk_indices)

    s_probs = F.softmax(student_topk_logits, dim=-1)
    s_log_probs = F.log_softmax(student_topk_logits, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits_at_topk_indices, dim=-1)

    return ((s_probs * (s_log_probs - t_log_probs)).sum(dim=-1) * mask).sum() / mask.sum().clamp(min=1.0)


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    hparams: OPSDHparams,
    collate_fn: Callable,
    rollout_fn: Callable,
    get_feedback_fn: Callable,
    validators: List[Tuple[Validator, ValidatorRunConfig]],
    output_dir: str = "outputs/opsd",
    wandb_project: str = "kdpo-exploration",
    wandb_run_name: str = "opsd",
) -> Dict[str, Any]:
    output_dir = f"{output_dir}/{wandb_run_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    world_size = get_world_size()
    adjusted_gradient_accumulation_steps = hparams.gradient_accumulation_steps // world_size

    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=adjusted_gradient_accumulation_steps,
        adjust_scheduler=True
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
                "teacher_alpha": hparams.teacher_alpha,
                "top_k": hparams.top_k,
                "temperature": hparams.temperature,
            },
            init_kwargs={
                "wandb": {"name": wandb_run_name}
            } if wandb_run_name else None
        )

    dataloader = DataLoader(
        dataset, batch_size=hparams.minibatch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay
    )

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader)

    teacher = EMATeacher(accelerator.unwrap_model(model),
                         alpha=hparams.teacher_alpha)

    model.train()

    global_step = 0
    total_loss = 0.0
    validation_history = defaultdict(list)

    if accelerator.is_main_process:
        logger.info("Beginning training...")
        logger.info(f"World size: {world_size}")
        logger.info(f"Learning rate: {hparams.learning_rate}")
        logger.info(f"Number of training steps: {len(dataloader)}")
        logger.info(f"Number of validation steps: {len(validators)}")
        logger.info(f"Number of epochs: {hparams.num_epochs}")
        logger.info(
            f"Gradient accumulation steps: {adjusted_gradient_accumulation_steps}")
        logger.info(f"Number of rollouts: {hparams.num_rollouts}")
        logger.info(f"Max prompt length: {hparams.max_prompt_length}")
        logger.info(f"Max response length: {hparams.max_response_length}")
        logger.info(f"Teacher alpha: {hparams.teacher_alpha}")
        logger.info(f"Top k: {hparams.top_k}")
        logger.info(f"Temperature: {hparams.temperature}")

    for epoch in range(hparams.num_epochs):
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1}/{hparams.num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            logger.info(
                f"Processing batch {batch_idx + 1} of {len(dataloader)}...")

            with accelerator.accumulate(model):
                batch_data = []

                for example in batch:
                    rollouts = rollout_fn(
                        accelerator.unwrap_model(model),
                        tokenizer,
                        example,
                        num_rollouts=hparams.num_rollouts,
                        temperature=hparams.temperature,
                        max_new_tokens=hparams.max_response_length
                    )
                    feedbacks = [get_feedback_fn(
                        rollout.completion, example) for rollout in rollouts]

                    batch_data.extend([{
                        "example": example,
                        "rollout": rollout,
                        "feedback": feedback
                    } for rollout, feedback in zip(rollouts, feedbacks)])

                student_logits, _, student_masks = get_logits_completion_ids_and_mask(
                    model,
                    tokenizer,
                    [build_student_messages(
                        get_question(data["example"]), data["rollout"].completion) for data in batch_data],
                    requires_grad=True
                )
                teacher_logits, _, _ = get_logits_completion_ids_and_mask(
                    teacher,
                    tokenizer,
                    [build_teacher_messages(get_question(data["example"]), data["rollout"].completion, data["feedback"].feedback_text,
                                            data["rollout"].completion) for data in batch_data],
                    requires_grad=False
                )

                loss = compute_loss(student_logits, teacher_logits,
                                    student_masks, hparams.top_k)

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

                    teacher.update(accelerator.unwrap_model(model))

                    global_step += 1
                    total_loss += loss.item()

                    if global_step % hparams.log_interval == 0 and accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step} - Loss: {loss.item():.4f}")

                        log_dict = {
                            "train/loss": loss.item(),
                            "train/global_step": global_step,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/completion_tokens": student_masks.sum().item(),
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

                        for validator, val_config in validators:
                            logger.info(f"Validating with {validator.name}...")
                            try:
                                score = validator.validate(
                                    model=unwrapped_model,
                                    tokenizer=tokenizer,
                                    batch_size=val_config.batch_size,
                                    max_new_tokens=val_config.max_new_tokens,
                                    max_seq_length=val_config.max_seq_length
                                )
                                validation_history[validator.name].append({
                                    "step": global_step,
                                    "score": score
                                })
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

        for validator, val_config in validators:
            logger.info(f"Validating with {validator.name}...")
            try:
                score = validator.validate(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    batch_size=val_config.batch_size,
                    max_new_tokens=val_config.max_new_tokens,
                    max_seq_length=val_config.max_seq_length
                )
                validation_history[validator.name].append({
                    "step": "final",
                    "score": score
                })
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
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--teacher-alpha", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/opsd")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    hparams = OPSDHparams(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_rollouts=args.num_rollouts,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        teacher_alpha=args.teacher_alpha,
        top_k=args.top_k,
        temperature=args.temperature,
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
