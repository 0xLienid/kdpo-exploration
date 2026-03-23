import logging
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.livecodebench import format_question as get_question
from training.train import Hparams, ValidatorRunConfig, train as run_train
from training.utils import (
    build_reference_model,
    build_student_messages,
    build_teacher_messages,
    build_teacher_prompt,
    gather_completion_span,
    get_logits_completion_ids_and_mask,
)


logger = logging.getLogger(__name__)


def compute_rollout_overlap(student_completion: str, teacher_completion: str) -> float:
    student_tokens = re.findall(r"\S+", student_completion)
    teacher_tokens = re.findall(r"\S+", teacher_completion)

    if not student_tokens and not teacher_tokens:
        return 1.0
    if not student_tokens or not teacher_tokens:
        return 0.0
    return SequenceMatcher(
        None,
        student_tokens,
        teacher_tokens,
        autojunk=False,
    ).ratio()


@dataclass
class KDPOHparams(Hparams):
    max_steps_per_epoch: Optional[int] = 40
    num_rollouts: int = 4
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    top_k: int = 20
    temperature: float = 1.0
    alpha: float = 0.5


def _seq_logprobs(logits, token_ids, lengths, max_len):
    logits = logits[:, :max_len]
    token_ids = token_ids[:, :max_len]
    positions = torch.arange(max_len, device=logits.device).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    token_logprobs = (
        torch.gather(logits, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
        - torch.logsumexp(logits, dim=-1)
    )
    return (token_logprobs * mask).sum(dim=1) / lengths.clamp(min=1)


def compute_loss(
    student_logits_y: torch.Tensor,
    student_logits_y_hat: torch.Tensor,
    completion_ids_y: torch.Tensor,
    completion_ids_y_hat: torch.Tensor,
    teacher_logits_y: torch.Tensor,
    teacher_logits_y_hat: torch.Tensor,
    student_lengths_y: torch.Tensor,
    student_lengths_y_hat: torch.Tensor,
    teacher_lengths_y: torch.Tensor,
    teacher_lengths_y_hat: torch.Tensor,
    k: int = 20,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # --- OPSD: top-k KL_y(π_s || π_T) ---
    opsd_lengths = torch.stack([student_lengths_y, teacher_lengths_y], dim=0).amin(dim=0)
    max_opsd = int(opsd_lengths.max().item())

    if max_opsd == 0:
        return student_logits_y.sum() * 0.0, {
            "y_kl": 0.0,
            "reward": 0.0,
            "pref_loss": 0.0,
        }

    opsd_positions = torch.arange(max_opsd, device=student_logits_y.device).unsqueeze(0)
    opsd_mask = opsd_positions < opsd_lengths.unsqueeze(1)

    s_logits_y = student_logits_y[:, :max_opsd]
    t_logits_y = teacher_logits_y[:, :max_opsd]

    student_topk_logits_y, student_topk_indices_y = torch.topk(s_logits_y, k, dim=-1)
    teacher_logits_y_at_topk = torch.gather(t_logits_y, dim=-1, index=student_topk_indices_y)

    student_topk_logprobs_y = student_topk_logits_y - torch.logsumexp(student_topk_logits_y, dim=-1, keepdim=True)
    teacher_topk_logprobs_y = teacher_logits_y_at_topk - torch.logsumexp(teacher_logits_y_at_topk, dim=-1, keepdim=True)
    student_topk_probs_y = student_topk_logprobs_y.exp()
    kl_y = (student_topk_probs_y * (student_topk_logprobs_y - teacher_topk_logprobs_y)).sum(dim=-1)
    del student_topk_logits_y, teacher_logits_y_at_topk, student_topk_probs_y, teacher_topk_logprobs_y

    kl_loss_y = (kl_y * opsd_mask).sum(dim=1) / opsd_lengths.clamp(min=1)

    # --- Sequence logprobs for reward and preference ---
    y_lengths = opsd_lengths
    y_hat_lengths = torch.stack([student_lengths_y_hat, teacher_lengths_y_hat], dim=0).amin(dim=0)
    max_y = max_opsd
    max_y_hat = int(y_hat_lengths.max().item())

    teacher_seq_logprob_y = _seq_logprobs(teacher_logits_y, completion_ids_y, y_lengths, max_y)
    teacher_seq_logprob_y_hat = _seq_logprobs(teacher_logits_y_hat, completion_ids_y_hat, y_hat_lengths, max_y_hat)
    student_seq_logprob_y = _seq_logprobs(student_logits_y, completion_ids_y, y_lengths, max_y)
    student_seq_logprob_y_hat = _seq_logprobs(student_logits_y_hat, completion_ids_y_hat, y_hat_lengths, max_y_hat)

    # R = value of feedback: how much does the feedback context boost y' relative to y?
    # feedback_boost_y_hat = teacher_seq_logprob_y_hat - student_seq_logprob_y_hat
    # feedback_boost_y = teacher_seq_logprob_y - student_seq_logprob_y
    # R = (feedback_boost_y_hat - feedback_boost_y).detach().clamp(min=0.0)

    # --- Preference: -R · (log π_s(y')/|y'| - log π_s(y)/|y|) ---
    pref_loss = -(student_seq_logprob_y_hat - student_seq_logprob_y)

    loss = (kl_loss_y + alpha * pref_loss).mean()

    metrics = {
        "y_kl": kl_y.mean().item(),
        "reward": R.mean().item(),
        "pref_loss": pref_loss.mean().item(),
        "feedback_boost_y_hat": feedback_boost_y_hat.mean().item(),
        "feedback_boost_y": feedback_boost_y.mean().item(),
    }
    return loss, metrics


def teacher_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Dict[str, Any],
    rollouts: List[Any],
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

    question = f"{example['question_title']}:\n{example['question_content']}"
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


def forward(
    accelerator,
    model: AutoModelForCausalLM,
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    hparams: KDPOHparams,
    auxiliary_model: Any,
    rollout_fn: Callable,
    get_feedback_fn: Callable,
) -> Optional[Dict[str, Any]]:
    batch_data = []
    overlap_scores = []
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
        feedbacks = [get_feedback_fn(rollout["completion"], example) for rollout in rollouts]
        student_reward_pct = sum(1.0 if feedback.success else 0.0 for feedback in feedbacks) / len(feedbacks)

        teacher_rollouts = teacher_rollout(
            unwrapped_model,
            tokenizer,
            example,
            rollouts,
            feedbacks,
            temperature=hparams.temperature,
            max_new_tokens=hparams.max_response_length,
        )
        teacher_feedbacks = [get_feedback_fn(teacher_regen["completion"], example) for teacher_regen in teacher_rollouts]
        teacher_reward_pct = sum(1.0 if feedback.success else 0.0 for feedback in teacher_feedbacks) / len(teacher_feedbacks)
        overlap_scores.append(
            sum(
                compute_rollout_overlap(rollout["completion"], teacher_regen["completion"])
                for rollout, teacher_regen in zip(rollouts, teacher_rollouts)
            ) / len(rollouts)
        )

        batch_data.extend(
            {
                "example": example,
                "rollout": rollout,
                "feedback": feedback,
                "teacher_rollout": teacher_regen,
            }
            for rollout, feedback, teacher_regen in zip(
                rollouts, feedbacks, teacher_rollouts
            )
        )

    if not batch_data:
        return None

    model.train()

    student_messages = [
        build_student_messages(get_question(data["example"]), data["rollout"]["completion"])
        for data in batch_data
    ] + [
        build_student_messages(
            get_question(data["example"]),
            data["teacher_rollout"]["completion"],
        )
        for data in batch_data
    ]
    student_logits, completion_ids, student_starts, student_lengths = (
        get_logits_completion_ids_and_mask(
            model,
            tokenizer,
            student_messages,
            requires_grad=True,
        )
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
    del student_logits, completion_ids, student_starts, student_lengths

    teacher_messages = [
        build_teacher_messages(
            get_question(data["example"]),
            data["rollout"]["completion"],
            data["feedback"].feedback_text,
            data["rollout"]["completion"],
        )
        for data in batch_data
    ] + [
        build_teacher_messages(
            get_question(data["example"]),
            data["rollout"]["completion"],
            data["feedback"].feedback_text,
            data["teacher_rollout"]["completion"],
        )
        for data in batch_data
    ]
    teacher_logits, _, teacher_starts, teacher_lengths = get_logits_completion_ids_and_mask(
        model,
        tokenizer,
        teacher_messages,
        requires_grad=False,
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
    del teacher_logits, teacher_starts, teacher_lengths

    completion_tokens = torch.stack(
        [student_lengths_y, student_lengths_y_hat, teacher_lengths_y, teacher_lengths_y_hat],
        dim=0,
    ).amin(dim=0).sum().item()
    student_teacher_rollout_overlap = sum(overlap_scores) / len(overlap_scores)

    return {
        "student_logits_y": student_logits_y,
        "student_logits_y_hat": student_logits_y_hat,
        "completion_ids_y": completion_ids_y,
        "completion_ids_y_hat": completion_ids_y_hat,
        "teacher_logits_y": teacher_logits_y,
        "teacher_logits_y_hat": teacher_logits_y_hat,
        "student_lengths_y": student_lengths_y,
        "student_lengths_y_hat": student_lengths_y_hat,
        "teacher_lengths_y": teacher_lengths_y,
        "teacher_lengths_y_hat": teacher_lengths_y_hat,
        "completion_tokens": completion_tokens,
        "student_reward_pct": student_reward_pct,
        "teacher_reward_pct": teacher_reward_pct,
        "student_teacher_rollout_overlap": student_teacher_rollout_overlap,
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
        hparams: KDPOHparams,
        auxiliary_model: Any,
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

        loss, metrics = compute_loss(
            student_logits_y=forward_outputs["student_logits_y"],
            student_logits_y_hat=forward_outputs["student_logits_y_hat"],
            completion_ids_y=forward_outputs["completion_ids_y"],
            completion_ids_y_hat=forward_outputs["completion_ids_y_hat"],
            teacher_logits_y=forward_outputs["teacher_logits_y"],
            teacher_logits_y_hat=forward_outputs["teacher_logits_y_hat"],
            student_lengths_y=forward_outputs["student_lengths_y"],
            student_lengths_y_hat=forward_outputs["student_lengths_y_hat"],
            teacher_lengths_y=forward_outputs["teacher_lengths_y"],
            teacher_lengths_y_hat=forward_outputs["teacher_lengths_y_hat"],
            k=hparams.top_k,
            alpha=hparams.alpha,
        )
        metrics["completion_tokens"] = forward_outputs["completion_tokens"]
        metrics["student_reward_pct"] = forward_outputs["student_reward_pct"]
        metrics["teacher_reward_pct"] = forward_outputs["teacher_reward_pct"]
        metrics["student_teacher_rollout_overlap"] = forward_outputs["student_teacher_rollout_overlap"]
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
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
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
        max_steps_per_epoch=args.max_steps_per_epoch,
        minibatch_size=args.minibatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_rollouts=args.num_rollouts,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        top_k=args.top_k,
        temperature=args.temperature,
        alpha=args.alpha,
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
        auxiliary_model=reference_model,
        validators=validators,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )