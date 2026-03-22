import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def teacher_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Dict[str, Any],
    student_rollout: Dict[str, Any],
    feedback: Any,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> Dict[str, Any]:
    from data.livecodebench import format_question
    from training.utils import build_teacher_prompt

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)
    if gc_was_enabled:
        model.gradient_checkpointing_disable()

    question = format_question(example)
    prompt = build_teacher_prompt(
        question,
        student_rollout["completion"],
        feedback.feedback_text,
    )
    inputs = tokenizer.apply_chat_template(
        [prompt],
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_tensors="pt",
        return_in_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

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

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    completion = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    tokenizer.padding_side = original_padding_side
    if gc_was_enabled:
        model.gradient_checkpointing_enable()

    return {"prompt": prompt, "completion": completion}


def get_completion_logits_for_messages(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, int]:
    from training.utils import get_logits_completion_ids_and_mask

    logits, _, completion_starts, completion_lengths = get_logits_completion_ids_and_mask(
        model,
        tokenizer,
        [messages],
        requires_grad=False,
    )

    completion_start = int(completion_starts[0].item())
    completion_length = int(completion_lengths[0].item())
    completion_logits = logits[0, completion_start:completion_start + completion_length].clone()

    del logits, completion_starts, completion_lengths
    gc.collect()
    maybe_empty_cuda_cache()

    return completion_logits, completion_length


def compute_full_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    vocab_chunk_size: int,
) -> torch.Tensor:
    student_log_z = torch.logsumexp(student_logits.float(), dim=-1)
    teacher_log_z = torch.logsumexp(teacher_logits.float(), dim=-1)

    full_reverse_kl = torch.zeros(
        student_logits.shape[0],
        device=student_logits.device,
        dtype=torch.float32,
    )
    vocab_size = student_logits.shape[-1]

    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        student_chunk = student_logits[:, start:end].float()
        teacher_chunk = teacher_logits[:, start:end].float()
        student_logprobs_chunk = student_chunk - student_log_z.unsqueeze(-1)
        teacher_logprobs_chunk = teacher_chunk - teacher_log_z.unsqueeze(-1)
        full_reverse_kl += (
            student_logprobs_chunk.exp()
            * (student_logprobs_chunk - teacher_logprobs_chunk)
        ).sum(dim=-1)

    return full_reverse_kl


def compute_teacher_rollout_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int,
    vocab_chunk_size: int,
) -> Dict[str, torch.Tensor]:
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()

    top_k = min(top_k, teacher_logits.shape[-1])
    student_log_z = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    teacher_log_z = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)

    teacher_topk_logits, teacher_topk_indices = torch.topk(
        teacher_logits,
        k=top_k,
        dim=-1,
    )
    student_logits_at_teacher_topk = torch.gather(
        student_logits,
        dim=-1,
        index=teacher_topk_indices,
    )

    restricted_student_logprobs = student_logits_at_teacher_topk - torch.logsumexp(
        student_logits_at_teacher_topk,
        dim=-1,
        keepdim=True,
    )
    restricted_teacher_logprobs = teacher_topk_logits - torch.logsumexp(
        teacher_topk_logits,
        dim=-1,
        keepdim=True,
    )
    restricted_student_probs = restricted_student_logprobs.exp()
    topk_reverse_kl = (
        restricted_student_probs
        * (restricted_student_logprobs - restricted_teacher_logprobs)
    ).sum(dim=-1)

    student_logprobs_on_teacher_topk = student_logits_at_teacher_topk - student_log_z
    teacher_logprobs_on_teacher_topk = teacher_topk_logits - teacher_log_z
    average_student_probs_on_teacher_topk = student_logprobs_on_teacher_topk.exp().mean(dim=-1)
    average_student_logprobs_on_teacher_topk = student_logprobs_on_teacher_topk.mean(dim=-1)
    average_teacher_logprobs_on_teacher_topk = teacher_logprobs_on_teacher_topk.mean(dim=-1)

    full_reverse_kl = compute_full_reverse_kl(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        vocab_chunk_size=vocab_chunk_size,
    )

    return {
        "topk_reverse_kl": topk_reverse_kl,
        "full_reverse_kl": full_reverse_kl,
        "average_student_probs_on_teacher_topk": average_student_probs_on_teacher_topk,
        "average_student_logprobs_on_teacher_topk": average_student_logprobs_on_teacher_topk,
        "average_teacher_logprobs_on_teacher_topk": average_teacher_logprobs_on_teacher_topk,
    }


def build_ventile_summary(
    metrics: Dict[str, torch.Tensor],
    sequence_length: int,
    num_windows: int = 20,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []

    for window_idx in range(num_windows):
        start = math.floor(window_idx * sequence_length / num_windows)
        end = math.floor((window_idx + 1) * sequence_length / num_windows)

        summary: Dict[str, Any] = {
            "ventile_index": window_idx,
            "start_token": start,
            "end_token_exclusive": end,
            "token_count": end - start,
            "start_pct": round((100.0 * window_idx) / num_windows, 2),
            "end_pct": round((100.0 * (window_idx + 1)) / num_windows, 2),
        }

        if end <= start:
            summary.update(
                {
                    "topk_reverse_kl": None,
                    "full_reverse_kl": None,
                    "average_student_probs_on_teacher_topk": None,
                    "average_student_logprobs_on_teacher_topk": None,
                    "average_teacher_logprobs_on_teacher_topk": None,
                }
            )
        else:
            for key, values in metrics.items():
                summary[key] = values[start:end].mean().item()

        summaries.append(summary)

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--student-max-new-tokens", type=int, default=2048)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--vocab-chunk-size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    from data.livecodebench import (
        LiveCodeBenchDataset,
        format_question,
        get_environment_feedback,
        rollout,
    )
    from training.utils import build_student_messages, build_teacher_messages

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()

    dataset = LiveCodeBenchDataset()
    example = dataset[args.example_index]
    question = format_question(example)

    student_rollouts = rollout(
        model,
        tokenizer,
        example,
        num_rollouts=1,
        temperature=args.student_temperature,
        max_new_tokens=args.student_max_new_tokens,
    )
    student_rollout_result = student_rollouts[0]
    feedback = get_environment_feedback(student_rollout_result["completion"], example)

    teacher_rollout_result = teacher_rollout(
        model,
        tokenizer,
        example,
        student_rollout_result,
        feedback,
        temperature=args.teacher_temperature,
        max_new_tokens=args.teacher_max_new_tokens,
    )
    teacher_feedback = get_environment_feedback(teacher_rollout_result["completion"], example)

    student_messages = build_student_messages(
        question,
        teacher_rollout_result["completion"],
    )
    teacher_messages = build_teacher_messages(
        question,
        student_rollout_result["completion"],
        feedback.feedback_text,
        teacher_rollout_result["completion"],
    )

    student_completion_logits, student_completion_length = get_completion_logits_for_messages(
        model,
        tokenizer,
        student_messages,
    )
    teacher_completion_logits, teacher_completion_length = get_completion_logits_for_messages(
        model,
        tokenizer,
        teacher_messages,
    )

    common_length = min(student_completion_length, teacher_completion_length)
    if common_length == 0:
        raise RuntimeError("No shared teacher-rollout completion tokens were available for analysis.")

    metrics = compute_teacher_rollout_metrics(
        student_logits=student_completion_logits[:common_length],
        teacher_logits=teacher_completion_logits[:common_length],
        top_k=args.top_k,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    ventiles = build_ventile_summary(
        metrics=metrics,
        sequence_length=common_length,
        num_windows=20,
    )

    output = {
        "model_name": args.model_name,
        "example_index": args.example_index,
        "top_k": args.top_k,
        "vocab_chunk_size": args.vocab_chunk_size,
        "question": question,
        "student_rollout": {
            "completion": student_rollout_result["completion"],
            "feedback_text": feedback.feedback_text,
            "success": feedback.success,
        },
        "teacher_rollout": {
            "completion": teacher_rollout_result["completion"],
            "feedback_text": teacher_feedback.feedback_text,
            "success": teacher_feedback.success,
        },
        "analysis": {
            "student_completion_length": student_completion_length,
            "teacher_completion_length": teacher_completion_length,
            "common_length": common_length,
            "global_means": {
                key: value.mean().item()
                for key, value in metrics.items()
            },
            "ventiles": ventiles,
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
