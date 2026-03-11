import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


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
        truncation=True,
        max_length=max_seq_length,
    ).to(device)
    full_encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    ).to(device)

    prompt_lengths = prompt_encodings["attention_mask"].sum(dim=-1)
    sequence_lengths = full_encodings["attention_mask"].sum(dim=-1)

    if requires_grad:
        outputs = model(**full_encodings)
    else:
        with torch.no_grad():
            outputs = model(**full_encodings)

    logits = outputs.logits.float()
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = full_encodings["input_ids"][:, 1:]

    token_logprobs = F.log_softmax(shifted_logits, dim=-1)
    token_logprobs = torch.gather(
        token_logprobs, dim=-1, index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)

    positions = torch.arange(
        full_encodings["input_ids"].shape[-1], device=device
    ).unsqueeze(0)
    completion_mask = (
        (positions >= prompt_lengths.unsqueeze(1))
        & (positions < sequence_lengths.unsqueeze(1))
    )
    token_mask = completion_mask[:, 1:] & (full_encodings["attention_mask"][:, 1:] > 0)

    return token_logprobs, token_mask


def get_logits_completion_ids_and_mask(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[List[Dict[str, Any]]],
    requires_grad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_lengths = [len(tokenizer.apply_chat_template(
        [conversation[0]],
        add_generation_prompt=True,
        tokenize=True
    )) for conversation in messages]

    full_encodings = tokenizer.apply_chat_template(
        full_messages, tokenize=True, padding=True, return_tensors="pt", return_in_dict=True)
    completion_lengths = full_encodings["attention_mask"].sum(
        dim=-1) - torch.tensor(prompt_lengths)

    full_encodings = {k: v.to(model.device) for k, v in full_encodings.items()}

    if requires_grad:
        outputs = model(**full_encodings)
    else:
        with torch.no_grad():
            outputs = model(**full_encodings)

    max_completion_length = completion_lengths.max().item()
    logits = torch.zeros(
        (outputs.logits.shape[0], max_completion_length, outputs.logits.shape[-1]), device=model.device)
    completion_ids = torch.full(
        (outputs.logits.shape[0], max_completion_length), tokenizer.pad_token_id, device=model.device)
    mask = torch.zeros(
        (outputs.logits.shape[0], max_completion_length), device=model.device)

    for i in range(len(messages)):
        end_idx = prompt_lengths[i] + completion_lenghs[i] - 1
        logits[i, :completion_lengths[i], :] = outputs.logits[i,
                                                              prompt_lengths[i] - 1:end_idx, :]
        completion_ids[i, :completion_lengths[i]] = full_encodings["input_ids"][i,
                                                                                prompt_lengths[i]:end_idx + 1].to(torch.int64)
        mask[i, :completion_lengths[i]] = full_encodings["attention_mask"][i,
                                                                           prompt_lengths[i]:end_idx + 1]

    return logits, completion_ids, mask


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
