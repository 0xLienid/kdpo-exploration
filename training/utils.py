import os
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_grad_norm(model: torch.nn.Module) -> float:
    total_sq_norm = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total_sq_norm += grad_norm * grad_norm
    return math.sqrt(total_sq_norm)


def gather_completion_span(
    tensor: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: int | float = 0,
) -> torch.Tensor:
    max_length = int(lengths.max().item())
    if max_length == 0:
        return tensor[:, :0]

    relative_positions = torch.arange(
        max_length, device=tensor.device).unsqueeze(0)
    batch_indices = torch.arange(
        tensor.shape[0], device=tensor.device).unsqueeze(1)
    positions = starts.unsqueeze(1) + relative_positions
    positions = positions.clamp(max=tensor.shape[1] - 1)

    span = tensor[batch_indices, positions]
    valid_mask = relative_positions < lengths.unsqueeze(1)
    if span.ndim == 3:
        span = span.masked_fill(~valid_mask.unsqueeze(-1), pad_value)
    else:
        span = span.masked_fill(~valid_mask, pad_value)

    return span


def get_completion_token_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[List[Dict[str, Any]]],
    max_seq_length: int,
    requires_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device

    prompt_texts = [
        tokenizer.apply_chat_template(
            [conversation[0]],
            add_generation_prompt=True,
            tokenize=False,
        )
        for conversation in messages
    ]
    full_texts = [
        tokenizer.apply_chat_template(conversation, tokenize=False)
        for conversation in messages
    ]

    prompt_encodings = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
    ).to(device)
    full_encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    prompt_lengths = prompt_encodings["attention_mask"].sum(dim=-1)
    sequence_lengths = full_encodings["attention_mask"].sum(dim=-1)

    if requires_grad:
        outputs = model(**full_encodings, use_cache=False)
    else:
        with torch.no_grad():
            outputs = model(**full_encodings, use_cache=False)

    logits = outputs.logits
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = full_encodings["input_ids"][:, 1:]

    min_completion_start = int((prompt_lengths - 1).min().item())
    max_completion_end = int((sequence_lengths - 1).max().item())
    shifted_logits = shifted_logits[:, min_completion_start:max_completion_end, :]
    shifted_input_ids = shifted_input_ids[:, min_completion_start:max_completion_end]

    shifted_completion_logits = torch.gather(
        shifted_logits, dim=-1, index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)
    token_logprobs = shifted_completion_logits - \
        torch.logsumexp(shifted_logits, dim=-1)

    relative_positions = torch.arange(
        token_logprobs.shape[1], device=device
    ).unsqueeze(0)
    completion_starts = (prompt_lengths - 1 - min_completion_start).clamp(min=0)
    completion_lengths = (sequence_lengths - prompt_lengths).clamp(min=0)
    token_mask = (
        (relative_positions >= completion_starts.unsqueeze(1))
        & (relative_positions < (completion_starts + completion_lengths).unsqueeze(1))
    )

    return token_logprobs, token_mask


def get_logits_completion_ids_and_mask(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[List[Dict[str, Any]]],
    requires_grad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_lengths = torch.tensor([len(tokenizer.apply_chat_template(
        [conversation[0]],
        add_generation_prompt=True,
        tokenize=True
    )["input_ids"]) for conversation in messages], device=model.device, dtype=torch.long)

    full_encodings = tokenizer.apply_chat_template(
        messages, tokenize=True, padding=True, return_tensors="pt", return_in_dict=True)
    full_encodings = {k: v.to(model.device) for k, v in full_encodings.items()}
    input_ids = full_encodings["input_ids"]
    attention_mask = full_encodings["attention_mask"]
    sequence_lengths = attention_mask.sum(dim=-1)

    if requires_grad:
        outputs = model(**full_encodings, use_cache=False)
    else:
        with torch.inference_mode():
            outputs = model(**full_encodings, use_cache=False)

    logits = outputs.logits[:, :-1, :]
    completion_ids = input_ids[:, 1:].to(torch.int64)
    completion_starts = prompt_lengths - 1
    completion_lengths = sequence_lengths - prompt_lengths

    return logits, completion_ids, completion_starts, completion_lengths


def build_student_prompt(question) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": f"Answer the following question, please keep your reasoning concise, and put your code in a ```python{{code}}``` block:\n\n{question}"}
    ]


def build_student_messages(question: str, completion: str) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": f"Answer the following question, please keep your reasoning concise, and put your code in a ```python{{code}}``` block:\n\n{question}"},
        {"role": "assistant", "content": completion}
    ]


def build_teacher_prompt(question: str, student_attempt: str, feedback: str) -> List[Dict[str, Any]]:
    return [
        {"role": "user",
            "content": f"## Question\n{question}\n\n## Previous Attempt\n{student_attempt}\n\n## Feedback (from environment) for the previous attempt\n{feedback}\n\nCorrectly solve the original question."}
    ]


def build_teacher_messages(question: str, student_attempt: str, feedback: str, completion: str) -> List[Dict[str, Any]]:
    return [
        {"role": "user",
            "content": f"## Question\n{question}\n\n## Previous Attempt\n{student_attempt}\n\n## Feedback (from environment) for the previous attempt\n{feedback}\n\nCorrectly solve the original question."},
        {"role": "assistant", "content": completion}
    ]
