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
TeacherMessagesFn = Callable[[Dict[str, Any], str, Any, str], List[Dict[str, Any]]]


@dataclass
class OPSDHparams(Hparams):
    num_rollouts: int = 8
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    teacher_alpha: float = 0.01
    top_k: int = 20
    temperature: float = 1.0


def compute_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    k: int = 20,
) -> torch.Tensor:
    student_topk_logits, student_topk_indices = torch.topk(
        student_logits, k, dim=-1)
    teacher_logits_at_topk_indices = torch.gather(
        teacher_logits, dim=-1, index=student_topk_indices)

    s_probs = F.softmax(student_topk_logits, dim=-1)
    s_log_probs = F.log_softmax(student_topk_logits, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits_at_topk_indices, dim=-1)

    return ((s_probs * (s_log_probs - t_log_probs)).sum(dim=-1) * mask).sum() / mask.sum().clamp(min=1.0)


def forward(
    accelerator,
    model: AutoModelForCausalLM,
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    hparams: OPSDHparams,
    teacher: EMATeacher,
    rollout_fn: RolloutFn,
    get_feedback_fn: FeedbackFn,
    build_student_messages_fn: StudentMessagesFn,
    build_teacher_messages_fn: TeacherMessagesFn,
) -> Optional[Dict[str, Any]]:
    batch_data = []
    unwrapped_model = accelerator.unwrap_model(model)

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

        batch_data.extend(
            {
                "example": example,
                "rollout": rollout,
                "feedback": feedback,
            }
            for rollout, feedback in zip(rollouts, feedbacks)
        )

    if not batch_data:
        return None

    student_logits, _, student_completion_starts, student_completion_lengths = (
        get_logits_completion_ids_and_mask(
            model,
            tokenizer,
            [
                build_student_messages_fn(
                    data["example"], data["rollout"]["completion"])
                for data in batch_data
            ],
            requires_grad=True,
        )
    )
    student_logits = gather_completion_span(
        student_logits,
        student_completion_starts,
        student_completion_lengths,
    )

    teacher_logits, _, teacher_completion_starts, teacher_completion_lengths = (
        get_logits_completion_ids_and_mask(
            teacher,
            tokenizer,
            [
                build_teacher_messages_fn(
                    data["example"],
                    data["rollout"]["completion"],
                    data["feedback"].feedback_text,
                    data["rollout"]["completion"],
                )
                for data in batch_data
            ],
            requires_grad=False,
        )
    )
    teacher_logits = gather_completion_span(
        teacher_logits,
        teacher_completion_starts,
        teacher_completion_lengths,
    )

    relative_positions = torch.arange(
        student_logits.shape[1], device=student_logits.device
    ).unsqueeze(0)
    mask = relative_positions < student_completion_lengths.unsqueeze(1).clamp(
        max=student_logits.shape[1]
    )

    return {
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "mask": mask,
        "completion_tokens": mask.sum().item(),
    }


def make_forward_backward_fn(
    rollout_fn: RolloutFn,
    get_feedback_fn: FeedbackFn,
    build_student_messages_fn: StudentMessagesFn,
    build_teacher_messages_fn: TeacherMessagesFn,
) -> Callable:
    def forward_backward(
        accelerator,
        model: AutoModelForCausalLM,
        batch: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        hparams: OPSDHparams,
        auxiliary_model: EMATeacher,
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
            build_teacher_messages_fn,
        )
        if forward_outputs is None:
            return None, {}

        loss = compute_loss(
            student_logits=forward_outputs["student_logits"],
            teacher_logits=forward_outputs["teacher_logits"],
            mask=forward_outputs["mask"],
            k=hparams.top_k,
        )
        return loss, {"completion_tokens": forward_outputs["completion_tokens"]}

    return forward_backward


def on_optimizer_step(context: StepContext) -> None:
    if context.auxiliary_model is None:
        return
    context.auxiliary_model.update(context.accelerator.unwrap_model(context.model))
