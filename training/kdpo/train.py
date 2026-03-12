import os
import math
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
from training.common import ValidatorRunConfig
from training.utils import gather_completion_span, get_world_size, get_logits_completion_ids_and_mask, build_student_messages, build_teacher_prompt, build_teacher_messages
from validators import Validator

logger = logging.getLogger(__name__)


@dataclass
class KDPOHparams:
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    num_epochs: int = 1
    minibatch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_rollouts: int = 4

    max_prompt_length: int = 2048
    max_response_length: int = 8192

    top_k: int = 20
    temperature: float = 1.0
    beta: float = 0.5

    log_interval: int = 10
    validation_interval: int = 10


def get_grad_norm(model: torch.nn.Module) -> float:
    total_sq_norm = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total_sq_norm += grad_norm * grad_norm
    return math.sqrt(total_sq_norm)


def compute_loss(
    student_logits_y: torch.Tensor,
    student_logits_y_hat: torch.Tensor,
    completion_ids_y: torch.Tensor,
    teacher_logits_y: torch.Tensor,
    teacher_logits_y_hat: torch.Tensor,
    completion_ids_y_hat: torch.Tensor,
    student_lengths_y: torch.Tensor,
    student_lengths_y_hat: torch.Tensor,
    teacher_lengths_y: torch.Tensor,
    teacher_lengths_y_hat: torch.Tensor,
    k: int = 20,
    beta: float = 0.5
) -> torch.Tensor:
    common_lengths = torch.minimum(
        torch.minimum(student_lengths_y, student_lengths_y_hat),
        torch.minimum(teacher_lengths_y, teacher_lengths_y_hat),
    )
    max_common_length = int(common_lengths.max().item())
    if max_common_length == 0:
        return student_logits_y.sum() * 0.0

    relative_positions = torch.arange(
        max_common_length, device=student_logits_y.device).unsqueeze(0)
    valid_mask = relative_positions < common_lengths.unsqueeze(1)
    student_logits_y = student_logits_y[:, :max_common_length]
    student_logits_y_hat = student_logits_y_hat[:, :max_common_length]
    teacher_logits_y = teacher_logits_y[:, :max_common_length]
    teacher_logits_y_hat = teacher_logits_y_hat[:, :max_common_length]
    completion_ids_y = completion_ids_y[:, :max_common_length]
    completion_ids_y_hat = completion_ids_y_hat[:, :max_common_length]

    student_topk_logits_y, student_topk_indices_y = torch.topk(
        student_logits_y, k, dim=-1)
    teacher_logits_y_at_topk_indices = torch.gather(
        teacher_logits_y, dim=-1, index=student_topk_indices_y)

    teacher_topk_logits_y_hat, teacher_topk_indices_y_hat = torch.topk(
        teacher_logits_y_hat, k, dim=-1)
    student_logits_y_hat_at_topk_indices = torch.gather(
        student_logits_y_hat, dim=-1, index=teacher_topk_indices_y_hat)

    student_completion_logits_y = torch.gather(
        student_logits_y, dim=-1, index=completion_ids_y.unsqueeze(-1)).squeeze(-1)
    student_completion_logits_y_hat = torch.gather(
        student_logits_y_hat, dim=-1, index=completion_ids_y_hat.unsqueeze(-1)).squeeze(-1)
    teacher_completion_logits_y = torch.gather(
        teacher_logits_y, dim=-1, index=completion_ids_y.unsqueeze(-1)).squeeze(-1)
    teacher_completion_logits_y_hat = torch.gather(
        teacher_logits_y_hat, dim=-1, index=completion_ids_y_hat.unsqueeze(-1)).squeeze(-1)

    student_completion_logprobs_y = student_completion_logits_y - \
        torch.logsumexp(student_logits_y, dim=-1)
    student_completion_logprobs_y_hat = student_completion_logits_y_hat - \
        torch.logsumexp(student_logits_y_hat, dim=-1)
    teacher_completion_logprobs_y = teacher_completion_logits_y - \
        torch.logsumexp(teacher_logits_y, dim=-1)
    teacher_completion_logprobs_y_hat = teacher_completion_logits_y_hat - \
        torch.logsumexp(teacher_logits_y_hat, dim=-1)

    y_hat_relative_logprobs = student_completion_logprobs_y_hat - \
        teacher_completion_logprobs_y_hat
    y_relative_logprobs = student_completion_logprobs_y - teacher_completion_logprobs_y

    student_topk_logprobs_y_hat = student_logits_y_hat_at_topk_indices - \
        torch.logsumexp(student_logits_y_hat_at_topk_indices,
                        dim=-1, keepdim=True)
    teacher_topk_logprobs_y_hat = teacher_topk_logits_y_hat - \
        torch.logsumexp(teacher_topk_logits_y_hat, dim=-1, keepdim=True)
    student_topk_probs_y_hat = student_topk_logprobs_y_hat.exp()
    teacher_topk_probs_y_hat = teacher_topk_logprobs_y_hat.exp()

    log_m = torch.logaddexp(
        student_topk_logprobs_y_hat, teacher_topk_logprobs_y_hat) - math.log(2.0)
    kl_student_m_y_hat = (student_topk_probs_y_hat *
                          (student_topk_logprobs_y_hat - log_m)).sum(dim=-1)
    kl_teacher_m_y_hat = (teacher_topk_probs_y_hat *
                          (teacher_topk_logprobs_y_hat - log_m)).sum(dim=-1)
    js = 0.5 * (kl_student_m_y_hat + kl_teacher_m_y_hat)
    jss = 1 - (js / math.log(2.0))

    student_topk_logprobs_y = student_topk_logits_y - \
        torch.logsumexp(student_topk_logits_y, dim=-1, keepdim=True)
    teacher_topk_logprobs_y = teacher_logits_y_at_topk_indices - \
        torch.logsumexp(teacher_logits_y_at_topk_indices,
                        dim=-1, keepdim=True)
    student_topk_probs_y = student_topk_logprobs_y.exp()
    kl_y = (student_topk_probs_y * (student_topk_logprobs_y -
            teacher_topk_logprobs_y)).sum(dim=-1)

    token_scores = beta * \
        ((y_hat_relative_logprobs * jss) - (y_relative_logprobs * kl_y))
    token_loss = -F.logsigmoid(token_scores)
    valid_mask = valid_mask.to(token_loss.dtype)
    return (token_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)


def teacher_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Dict[str, Any],
    rollouts: List[Any],
    feedbacks: List[Any],
    temperature: float = 1.0,
    max_new_tokens: int = 2048
) -> List[Dict[str, Any]]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gc_was_enabled = model.is_gradient_checkpointing
    if gc_was_enabled:
        model.gradient_checkpointing_disable()

    question = f"{example['question_title']}:\n{example['question_content']}"
    prompts = [build_teacher_prompt(question, rollout["completion"], feedback.feedback_text) for rollout, feedback in zip(rollouts, feedbacks)]
    inputs = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, padding=True, return_tensors="pt", return_in_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.95

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    completions = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results = [{
        "prompt": prompt,
        "completion": completion
    } for prompt, completion in zip(prompts, completions)]

    tokenizer.padding_side = original_padding_side

    if gc_was_enabled:
        model.gradient_checkpointing_enable()

    return results


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    hparams: KDPOHparams,
    collate_fn: Callable,
    rollout_fn: Callable,
    get_feedback_fn: Callable,
    validators: List[Tuple[Validator, ValidatorRunConfig]],
    output_dir: str = "outputs/kdpo",
    wandb_project: str = "kdpo-exploration",
    wandb_run_name: str = "kdpo",
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
                "top_k": hparams.top_k,
                "temperature": hparams.temperature,
                "beta": hparams.beta,
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
    model.train()

    global_step = 0
    total_loss = 0.0
    validation_history = defaultdict(list)

    if accelerator.is_main_process:
        logger.info("Beginning training...")
        logger.info(f"Learning rate: {hparams.learning_rate}")
        logger.info(f"Number of training steps: {len(dataloader)}")
        logger.info(f"Number of validation steps: {len(validators)}")
        logger.info(f"Number of epochs: {hparams.num_epochs}")
        logger.info(
            f"Gradient accumulation steps: {adjusted_gradient_accumulation_steps}")
        logger.info(f"Number of rollouts: {hparams.num_rollouts}")
        logger.info(f"Max prompt length: {hparams.max_prompt_length}")
        logger.info(f"Max response length: {hparams.max_response_length}")
        logger.info(f"Top k: {hparams.top_k}")
        logger.info(f"Temperature: {hparams.temperature}")
        logger.info(f"Beta: {hparams.beta}")

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
                        rollout["completion"], example) for rollout in rollouts]

                    teacher_rollouts = teacher_rollout(
                        accelerator.unwrap_model(model),
                        tokenizer,
                        example,
                        rollouts,
                        feedbacks,
                        temperature=hparams.temperature,
                        max_new_tokens=hparams.max_response_length
                    )

                    batch_data.extend([{
                        "example": example,
                        "rollout": rollout,
                        "feedback": feedback,
                        "teacher_rollout": teacher_rollout
                    } for rollout, feedback, teacher_rollout in zip(rollouts, feedbacks, teacher_rollouts)])

                student_messages = [build_student_messages(f"{data['example']['question_title']}:\n{data['example']['question_content']}", data["rollout"]["completion"]) for data in batch_data] + [
                    build_student_messages(f"{data['example']['question_title']}:\n{data['example']['question_content']}", data["teacher_rollout"]["completion"]) for data in batch_data]
                student_logits, completion_ids, student_starts, student_lengths = get_logits_completion_ids_and_mask(
                    model,
                    tokenizer,
                    student_messages,
                    requires_grad=True
                )

                student_logits_y = gather_completion_span(
                    student_logits[:len(batch_data)],
                    student_starts[:len(batch_data)],
                    student_lengths[:len(batch_data)],
                )
                student_logits_y_hat = gather_completion_span(
                    student_logits[len(batch_data):],
                    student_starts[len(batch_data):],
                    student_lengths[len(batch_data):],
                )
                completion_ids_y = gather_completion_span(
                    completion_ids[:len(batch_data)],
                    student_starts[:len(batch_data)],
                    student_lengths[:len(batch_data)],
                )
                completion_ids_y_hat = gather_completion_span(
                    completion_ids[len(batch_data):],
                    student_starts[len(batch_data):],
                    student_lengths[len(batch_data):],
                )
                student_lengths_y = student_lengths[:len(batch_data)]
                student_lengths_y_hat = student_lengths[len(batch_data):]

                del student_logits, completion_ids, student_starts

                teacher_messages = [build_teacher_messages(f"{data['example']['question_title']}:\n{data['example']['question_content']}", data["rollout"]["completion"], data["feedback"].feedback_text, data["rollout"]["completion"]) for data in batch_data] + [
                    build_teacher_messages(f"{data['example']['question_title']}:\n{data['example']['question_content']}", data["rollout"]["completion"], data["feedback"].feedback_text, data["teacher_rollout"]["completion"]) for data in batch_data]
                teacher_logits, _, teacher_starts, teacher_lengths = get_logits_completion_ids_and_mask(
                    model,
                    tokenizer,
                    teacher_messages,
                    requires_grad=False
                )

                teacher_logits_y = gather_completion_span(
                    teacher_logits[:len(batch_data)],
                    teacher_starts[:len(batch_data)],
                    teacher_lengths[:len(batch_data)],
                )
                teacher_logits_y_hat = gather_completion_span(
                    teacher_logits[len(batch_data):],
                    teacher_starts[len(batch_data):],
                    teacher_lengths[len(batch_data):],
                )
                teacher_lengths_y = teacher_lengths[:len(batch_data)]
                teacher_lengths_y_hat = teacher_lengths[len(batch_data):]

                del teacher_logits, teacher_starts

                loss = compute_loss(student_logits_y, student_logits_y_hat, completion_ids_y, teacher_logits_y,
                                    teacher_logits_y_hat, completion_ids_y_hat, student_lengths_y,
                                    student_lengths_y_hat, teacher_lengths_y, teacher_lengths_y_hat,
                                    hparams.top_k, hparams.beta)

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
                    total_loss += loss.item()

                    if global_step % hparams.log_interval == 0 and accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step} - Loss: {loss.item():.4f}")

                        log_dict = {
                            "train/loss": loss.item(),
                            "train/global_step": global_step,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/completion_tokens": torch.minimum(
                                torch.minimum(student_lengths_y, student_lengths_y_hat),
                                torch.minimum(teacher_lengths_y, teacher_lengths_y_hat),
                            ).sum().item(),
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

        try:
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
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/kdpo")
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

    hparams = KDPOHparams(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_rollouts=args.num_rollouts,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        top_k=args.top_k,
        temperature=args.temperature,
        beta=args.beta,
        log_interval=args.log_interval,
        validation_interval=args.validation_interval,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
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
