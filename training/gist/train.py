import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.opsd.EMATeacher import EMATeacher
from training.train import (
    Hparams,
    StepContext,
)
from training.utils import (
    gather_completion_span,
    get_logits_completion_ids_and_mask,
)


logger = logging.getLogger(__name__)

RolloutFn = Callable[..., List[Dict[str, Any]]]
FeedbackFn = Callable[[str, Dict[str, Any]], Any]
StudentMessagesFn = Callable[[Dict[str, Any], str], List[Dict[str, Any]]]
InsightPromptFn = Callable[[Dict[str, Any], str, Any], List[Dict[str, Any]]]
InsightTeacherMessagesFn = Callable[[Dict[str, Any], str, str], List[Dict[str, Any]]]


@dataclass
class GISTHparams(Hparams):
    max_steps_per_epoch: Optional[int] = 40
    num_rollouts: int = 4
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    top_k: int = 20
    temperature: float = 1.0
    teacher_alpha: float = 0.0
    beta: float = 0.1


def compute_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_ids: torch.Tensor,
    mask: torch.Tensor,
    k: int = 20,
    beta: float = 0.1,
) -> torch.Tensor:
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_token_log_probs = teacher_log_probs.gather(dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_token_log_probs = student_log_probs.gather(dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)

    # PG: raw sums, no length normalization
    rewards = ((teacher_token_log_probs - student_token_log_probs) * mask).sum(dim=-1)
    baseline = rewards.mean()
    advantages = rewards - baseline

    student_seq_log_probs = (student_token_log_probs * mask).sum(dim=-1)
    pg_loss = -(advantages.detach() * student_seq_log_probs).mean()

    # KL: per-token normalized (no length bias by construction)
    student_topk_logits, student_topk_indices = torch.topk(
        student_logits, k, dim=-1)
    teacher_logits_at_topk_indices = torch.gather(
        teacher_logits, dim=-1, index=student_topk_indices)

    s_probs = F.softmax(student_topk_logits, dim=-1)
    s_log_probs = F.log_softmax(student_topk_logits, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits_at_topk_indices, dim=-1)

    kl_loss = ((s_probs * (s_log_probs - t_log_probs)).sum(dim=-1) * mask).sum() / mask.sum().clamp(min=1.0)

    return kl_loss + beta * pg_loss


def insight_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Dict[str, Any],
    rollouts: List[Any],
    feedbacks: List[Any],
    build_insight_prompt_fn: InsightPromptFn,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        build_insight_prompt_fn(example, rollout["completion"], feedback)
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

    return results


def forward(
    accelerator,
    model: AutoModelForCausalLM,
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    hparams: GISTHparams,
    auxiliary_model: Any,
    rollout_fn: RolloutFn,
    get_feedback_fn: FeedbackFn,
    build_student_messages_fn: StudentMessagesFn,
    build_insight_prompt_fn: InsightPromptFn,
    build_insight_teacher_messages_fn: InsightTeacherMessagesFn,
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

        teacher_insights = insight_rollout(
            auxiliary_model,
            tokenizer,
            example,
            rollouts,
            feedbacks,
            build_insight_prompt_fn,
            temperature=hparams.temperature,
            max_new_tokens=hparams.max_response_length,
        )

        batch_data.extend(
            {
                "example": example,
                "rollout": rollout,
                "feedback": feedback,
                "teacher_insight": teacher_insight,
            }
            for rollout, feedback, teacher_insight in zip(rollouts, feedbacks, teacher_insights)
        )

    if not batch_data:
        return None

    model.train()
    student_messages = [
        build_student_messages_fn(data["example"], data["rollout"]["completion"])
        for data in batch_data
    ]
    teacher_messages = [
        build_insight_teacher_messages_fn(
            data["example"],
            data["teacher_insight"]["completion"],
            data["rollout"]["completion"],
        )
        for data in batch_data
    ]

    student_logits, completion_ids, student_starts, student_lengths = get_logits_completion_ids_and_mask(
        model,
        tokenizer,
        student_messages,
        requires_grad=True,
    )
    student_logits = gather_completion_span(
        student_logits,
        student_starts,
        student_lengths,
    )
    completion_ids = gather_completion_span(
        completion_ids,
        student_starts,
        student_lengths,
    )

    teacher_logits, _, teacher_starts, teacher_lengths = get_logits_completion_ids_and_mask(
        auxiliary_model,
        tokenizer,
        teacher_messages,
        requires_grad=False,
    )
    teacher_logits = gather_completion_span(
        teacher_logits,
        teacher_starts,
        teacher_lengths,
    )

    relative_positions = torch.arange(
        student_logits.shape[1], device=student_logits.device
    ).unsqueeze(0)
    mask = relative_positions < student_lengths.unsqueeze(1).clamp(
        max=student_logits.shape[1]
    )

    return {
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "completion_ids": completion_ids,
        "mask": mask,
        "completion_tokens": mask.sum().item(),
    }


def make_forward_backward_fn(
    rollout_fn: RolloutFn,
    get_feedback_fn: FeedbackFn,
    build_student_messages_fn: StudentMessagesFn,
    build_insight_prompt_fn: InsightPromptFn,
    build_insight_teacher_messages_fn: InsightTeacherMessagesFn,
) -> Callable:
    def forward_backward(
        accelerator,
        model: AutoModelForCausalLM,
        batch: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        hparams: GISTHparams,
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
            build_student_messages_fn,
            build_insight_prompt_fn,
            build_insight_teacher_messages_fn,
        )
        if forward_outputs is None:
            return None, {}

        loss = compute_loss(
            student_logits=forward_outputs["student_logits"],
            teacher_logits=forward_outputs["teacher_logits"],
            completion_ids=forward_outputs["completion_ids"],
            mask=forward_outputs["mask"],
            k=hparams.top_k,
            beta=hparams.beta,
        )
        return loss, {"completion_tokens": forward_outputs["completion_tokens"]}

    return forward_backward


def on_optimizer_step(context: StepContext) -> None:
    if context.auxiliary_model is None:
        return
    context.auxiliary_model.update(context.accelerator.unwrap_model(context.model))
