import logging
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
    build_teacher_prompt,
    get_completion_token_logprobs,
)


logger = logging.getLogger(__name__)


@dataclass
class SDPOHparams(Hparams):
    num_rollouts: int = 4
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    temperature: float = 1.0
    beta: float = 0.1


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


def build_reference_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    reference_model = copy.deepcopy(model)
    if hasattr(reference_model, "gradient_checkpointing_disable"):
        reference_model.gradient_checkpointing_disable()
    reference_model.requires_grad_(False)
    reference_model.eval()
    return reference_model


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


def forward(
    accelerator,
    model: AutoModelForCausalLM,
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    hparams: SDPOHparams,
    reference_model: AutoModelForCausalLM,
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

        batch_data.extend(
            {
                "question": get_question(example),
                "chosen_completion": teacher_regen["completion"],
                "rejected_completion": rollout["completion"],
            }
            for rollout, teacher_regen in zip(rollouts, teacher_rollouts)
        )

    if not batch_data:
        return None

    rejected_messages = [
        build_student_messages(data["question"], data["rejected_completion"])
        for data in batch_data
    ]
    chosen_messages = [
        build_student_messages(data["question"], data["chosen_completion"])
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
    policy_sequence_logprobs = (policy_token_logprobs * policy_mask).sum(dim=-1)
    reference_sequence_logprobs = (
        reference_token_logprobs * reference_mask
    ).sum(dim=-1)

    pair_count = len(batch_data)

    return {
        "policy_rejected": policy_sequence_logprobs[:pair_count],
        "policy_chosen": policy_sequence_logprobs[pair_count:],
        "reference_rejected": reference_sequence_logprobs[:pair_count],
        "reference_chosen": reference_sequence_logprobs[pair_count:],
        "completion_tokens": policy_mask.sum().item(),
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
        hparams: SDPOHparams,
        auxiliary_model: AutoModelForCausalLM,
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

        loss, metrics = compute_dpo_loss(
            policy_chosen_logprobs=forward_outputs["policy_chosen"],
            policy_rejected_logprobs=forward_outputs["policy_rejected"],
            reference_chosen_logprobs=forward_outputs["reference_chosen"],
            reference_rejected_logprobs=forward_outputs["reference_rejected"],
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
        max_steps_per_epoch=args.max_steps_per_epoch,
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
        validators=validators,
        auxiliary_model=reference_model,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
