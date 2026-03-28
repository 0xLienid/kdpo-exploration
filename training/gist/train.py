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
    build_insight_teacher_messages,
    build_insight_prompt,
    gather_completion_span,
    get_logits_completion_ids_and_mask,
)


logger = logging.getLogger(__name__)


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
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    question = f"{example['question_title']}:\n{example['question_content']}"
    prompts = [
        build_insight_prompt(question, rollout["completion"], feedback.feedback_text)
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

        teacher_insights = insight_rollout(
            auxiliary_model,
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
                "teacher_insight": teacher_insight,
            }
            for rollout, feedback, teacher_insight in zip(rollouts, feedbacks, teacher_insights)
        )

    if not batch_data:
        return None

    model.train()
    student_messages = [build_student_messages(get_question(data["example"]), data["rollout"]["completion"]) for data in batch_data]
    teacher_messages = [build_insight_teacher_messages(get_question(data["example"]), data["teacher_insight"]["completion"], data["rollout"]["completion"]) for data in batch_data]

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
    rollout_fn: Callable,
    get_feedback_fn: Callable,
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
    parser.add_argument("--teacher-alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/gist")
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

    hparams = GISTHparams(
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
        teacher_alpha=args.teacher_alpha,
        beta=args.beta,
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
                batch_size=8,
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