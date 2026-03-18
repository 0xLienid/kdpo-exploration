import logging
import math
from dataclasses import dataclass
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


@dataclass
class KDPOHparams(Hparams):
    max_steps_per_epoch: Optional[int] = 40
    num_rollouts: int = 4
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    top_k: int = 20
    temperature: float = 1.0
    beta: float = 0.5


def compute_loss(
    student_logits_y: torch.Tensor,
    student_logits_y_hat: torch.Tensor,
    completion_ids_y: torch.Tensor,
    teacher_logits_y: torch.Tensor,
    teacher_logits_y_hat: torch.Tensor,
    completion_ids_y_hat: torch.Tensor,
    reference_completion_logprobs_y: torch.Tensor,
    reference_completion_logprobs_y_hat: torch.Tensor,
    student_lengths_y: torch.Tensor,
    student_lengths_y_hat: torch.Tensor,
    teacher_lengths_y: torch.Tensor,
    teacher_lengths_y_hat: torch.Tensor,
    reference_lengths_y: torch.Tensor,
    reference_lengths_y_hat: torch.Tensor,
    k: int = 20,
    beta: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    common_lengths = torch.stack(
        [
            student_lengths_y,
            student_lengths_y_hat,
            teacher_lengths_y,
            teacher_lengths_y_hat,
            reference_lengths_y,
            reference_lengths_y_hat,
        ],
        dim=0,
    ).amin(dim=0)
    max_common_length = int(common_lengths.max().item())
    if max_common_length == 0:
        return student_logits_y.sum() * 0.0, {
            "core_scores": 0.0,
            "weight": 0.0,
            "y_hat_kl": 0.0,
            "y_kl": 0.0,
        }

    relative_positions = torch.arange(
        max_common_length, device=student_logits_y.device).unsqueeze(0)
    valid_mask = relative_positions < common_lengths.unsqueeze(1)
    student_logits_y = student_logits_y[:, :max_common_length]
    student_logits_y_hat = student_logits_y_hat[:, :max_common_length]
    teacher_logits_y = teacher_logits_y[:, :max_common_length]
    teacher_logits_y_hat = teacher_logits_y_hat[:, :max_common_length]
    reference_completion_logprobs_y = reference_completion_logprobs_y[:, :max_common_length]
    reference_completion_logprobs_y_hat = reference_completion_logprobs_y_hat[:, :max_common_length]
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

    student_completion_logprobs_y = student_completion_logits_y - torch.logsumexp(
        student_logits_y, dim=-1)
    student_completion_logprobs_y_hat = student_completion_logits_y_hat - torch.logsumexp(
        student_logits_y_hat, dim=-1)
    del student_logits_y, student_logits_y_hat

    student_relative_logprobs = student_completion_logprobs_y_hat - student_completion_logprobs_y
    reference_relative_logprobs = reference_completion_logprobs_y - reference_completion_logprobs_y_hat
    del student_completion_logits_y, student_completion_logits_y_hat, reference_completion_logprobs_y, reference_completion_logprobs_y_hat

    student_topk_logprobs_y_hat = student_logits_y_hat_at_topk_indices - torch.logsumexp(
        student_logits_y_hat_at_topk_indices, dim=-1, keepdim=True)
    teacher_topk_logprobs_y_hat = teacher_topk_logits_y_hat - torch.logsumexp(
        teacher_topk_logits_y_hat, dim=-1, keepdim=True)
    student_topk_probs_y_hat = student_topk_logprobs_y_hat.exp()
    kl_y_hat = (student_topk_probs_y_hat * (student_topk_logprobs_y_hat - teacher_topk_logprobs_y_hat)).sum(dim=-1)
    del student_logits_y_hat_at_topk_indices, teacher_topk_logits_y_hat, student_topk_logprobs_y_hat, teacher_topk_logprobs_y_hat, student_topk_probs_y_hat

    student_topk_logprobs_y = student_topk_logits_y - torch.logsumexp(
        student_topk_logits_y, dim=-1, keepdim=True)
    teacher_topk_logprobs_y = teacher_logits_y_at_topk_indices - torch.logsumexp(
        teacher_logits_y_at_topk_indices, dim=-1, keepdim=True)
    student_topk_probs_y = student_topk_logprobs_y.exp()
    kl_y = (student_topk_probs_y * (student_topk_logprobs_y -
            teacher_topk_logprobs_y)).sum(dim=-1)
    del student_topk_logits_y, teacher_logits_y_at_topk_indices, student_topk_probs_y, teacher_topk_logprobs_y

    preference_scores = beta * (student_relative_logprobs + reference_relative_logprobs)
    weight = torch.sigmoid(student_relative_logprobs.detach())
    core_scores = -F.logsigmoid(preference_scores)
    y_hat_score = weight * kl_y_hat
    y_score = (1 - weight) * kl_y
    token_loss = core_scores + y_hat_score + y_score
    valid_mask = valid_mask.to(token_loss.dtype)

    loss = (token_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    metrics = {
        "core_scores": core_scores.mean().item(),
        "weight": weight.mean().item(),
        "y_hat_kl": kl_y_hat.mean().item(),
        "y_kl": kl_y.mean().item(),
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

        teacher_rollouts = teacher_rollout(
            unwrapped_model,
            tokenizer,
            example,
            rollouts,
            feedbacks,
            temperature=hparams.temperature,
            max_new_tokens=hparams.max_response_length,
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

    reference_logits, _, reference_starts, reference_lengths = get_logits_completion_ids_and_mask(
        auxiliary_model,
        tokenizer,
        student_messages,
        requires_grad=False,
    )
    reference_logits_y = gather_completion_span(
        reference_logits[:len(batch_data)],
        reference_starts[:len(batch_data)],
        reference_lengths[:len(batch_data)],
    )
    reference_logits_y_hat = gather_completion_span(
        reference_logits[len(batch_data):],
        reference_starts[len(batch_data):],
        reference_lengths[len(batch_data):],
    )
    reference_lengths_y = reference_lengths[:len(batch_data)]
    reference_lengths_y_hat = reference_lengths[len(batch_data):]

    reference_completion_logits_y = torch.gather(
        reference_logits_y, dim=-1, index=completion_ids_y.unsqueeze(-1)).squeeze(-1)
    reference_completion_logits_y_hat = torch.gather(
        reference_logits_y_hat, dim=-1, index=completion_ids_y_hat.unsqueeze(-1)).squeeze(-1)

    reference_completion_logprobs_y = reference_completion_logits_y - torch.logsumexp(
        reference_logits_y, dim=-1)
    reference_completion_logprobs_y_hat = reference_completion_logits_y_hat - torch.logsumexp(
        reference_logits_y_hat, dim=-1)
    del reference_logits, reference_logits_y, reference_logits_y_hat, reference_starts, reference_lengths, reference_completion_logits_y, reference_completion_logits_y_hat

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
        [
            student_lengths_y,
            student_lengths_y_hat,
            teacher_lengths_y,
            teacher_lengths_y_hat,
            reference_lengths_y,
            reference_lengths_y_hat,
        ],
        dim=0,
    ).amin(dim=0).sum().item()

    return {
        "student_logits_y": student_logits_y,
        "student_logits_y_hat": student_logits_y_hat,
        "completion_ids_y": completion_ids_y,
        "teacher_logits_y": teacher_logits_y,
        "reference_completion_logprobs_y": reference_completion_logprobs_y,
        "reference_completion_logprobs_y_hat": reference_completion_logprobs_y_hat,
        "teacher_logits_y_hat": teacher_logits_y_hat,
        "completion_ids_y_hat": completion_ids_y_hat,
        "student_lengths_y": student_lengths_y,
        "student_lengths_y_hat": student_lengths_y_hat,
        "teacher_lengths_y": teacher_lengths_y,
        "teacher_lengths_y_hat": teacher_lengths_y_hat,
        "reference_lengths_y": reference_lengths_y,
        "reference_lengths_y_hat": reference_lengths_y_hat,
        "completion_tokens": completion_tokens,
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
            teacher_logits_y=forward_outputs["teacher_logits_y"],
            teacher_logits_y_hat=forward_outputs["teacher_logits_y_hat"],
            reference_completion_logprobs_y=forward_outputs["reference_completion_logprobs_y"],
            reference_completion_logprobs_y_hat=forward_outputs["reference_completion_logprobs_y_hat"],
            completion_ids_y_hat=forward_outputs["completion_ids_y_hat"],
            student_lengths_y=forward_outputs["student_lengths_y"],
            student_lengths_y_hat=forward_outputs["student_lengths_y_hat"],
            teacher_lengths_y=forward_outputs["teacher_lengths_y"],
            teacher_lengths_y_hat=forward_outputs["teacher_lengths_y_hat"],
            reference_lengths_y=forward_outputs["reference_lengths_y"],
            reference_lengths_y_hat=forward_outputs["reference_lengths_y_hat"],
            k=hparams.top_k,
            beta=hparams.beta,
        )
        metrics["completion_tokens"] = forward_outputs["completion_tokens"]
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
        max_steps_per_epoch=args.max_steps_per_epoch,
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
