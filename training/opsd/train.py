import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.livecodebench import format_question as get_question
from training.opsd.EMATeacher import EMATeacher
from training.train import (
    Hparams,
    StepContext,
    ValidatorRunConfig,
    train as run_train,
)
from training.utils import (
    build_student_messages,
    build_teacher_messages,
    gather_completion_span,
    get_logits_completion_ids_and_mask,
)


logger = logging.getLogger(__name__)


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
    rollout_fn: Callable,
    get_feedback_fn: Callable,
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
                build_student_messages(
                    get_question(data["example"]),
                    data["rollout"]["completion"],
                )
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
                build_teacher_messages(
                    get_question(data["example"]),
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
    rollout_fn: Callable,
    get_feedback_fn: Callable,
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

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
        max_steps_per_epoch=args.max_steps_per_epoch,
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.gradient_checkpointing_enable()

    teacher = EMATeacher(model, alpha=hparams.teacher_alpha)
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
        on_optimizer_step_fn=on_optimizer_step,
        auxiliary_model=teacher,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
