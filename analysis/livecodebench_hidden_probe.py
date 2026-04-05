import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.livecodebench import LiveCodeBenchDataset, format_question, get_environment_feedback
from experiments.common import configure_logging, seed_everything


DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = "analysis/outputs/livecodebench_hidden_probe"

# The user asked for a fixed number of optimization steps controlled by a
# hardcoded constant rather than a CLI flag.
NUM_PROBE_STEPS = 32

STUDENT_TEMPERATURE = 1.0
TEACHER_TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 2048
PROBE_LEARNING_RATE = 1e-3
PROBE_WEIGHT_DECAY = 0.0
PROBE_LOSS = "mse"


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


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_observation_messages(
    example: Dict[str, Any],
    student_attempt: str,
    observation_text: str,
) -> List[Dict[str, Any]]:
    # The request uses `(example, y, o)` for `h_o`. On LiveCodeBench the
    # natural `o` is the environment observation / feedback text for `y`.
    question = format_question(example)
    return [
        {
            "role": "user",
            "content": (
                f"## Question\n{question}\n\n"
                f"## Previous Attempt\n{student_attempt}\n\n"
                "## Task\nProvide the environment feedback for the previous attempt."
            ),
        },
        {
            "role": "assistant",
            "content": observation_text,
        },
    ]


def build_student_messages(
    example: Dict[str, Any],
    completion: str,
) -> List[Dict[str, Any]]:
    question = format_question(example)
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question, please keep your reasoning "
                "concise, and put your code in a ```python{code}``` block:\n\n"
                f"{question}"
            ),
        },
        {
            "role": "assistant",
            "content": completion,
        },
    ]


def build_teacher_messages(
    example: Dict[str, Any],
    student_attempt: str,
    feedback_text: str,
    completion: str,
) -> List[Dict[str, Any]]:
    question = format_question(example)
    return [
        {
            "role": "user",
            "content": (
                f"## Question\n{question}\n\n"
                f"## Previous Attempt\n{student_attempt}\n\n"
                "## Feedback (from environment) for the previous attempt\n"
                f"{feedback_text}\n\n"
                "Correctly solve the original question."
            ),
        },
        {
            "role": "assistant",
            "content": completion,
        },
    ]


def tokenize_messages(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = False,
) -> Dict[str, torch.Tensor]:
    return tokenizer.apply_chat_template(
        [messages],
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        padding=True,
        return_tensors="pt",
        return_in_dict=True,
    )


def top_p_sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled_logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    filtered_logits = scaled_logits.clone()
    filtered_logits.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf")),
    )
    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def autoregressive_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_messages: List[Dict[str, Any]],
    temperature: float,
    max_new_tokens: int,
    track_final_hidden_state: bool = False,
) -> tuple[str, torch.Tensor | None]:
    ensure_pad_token(tokenizer)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    encodings = tokenize_messages(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    current_input_ids = input_ids
    current_attention_mask = attention_mask
    past_key_values = None
    generated_token_ids: List[int] = []
    final_hidden_state = None
    last_processed_hidden_state = None
    eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        for step_idx in range(max_new_tokens):
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=track_final_hidden_state,
                return_dict=True,
            )

            if track_final_hidden_state:
                last_processed_hidden_state = outputs.hidden_states[-1][:, -1, :].detach().float()

            next_token = top_p_sample(
                outputs.logits[:, -1, :],
                temperature=temperature,
                top_p=TOP_P,
            )

            past_key_values = outputs.past_key_values
            next_token_id = int(next_token.item())

            if eos_token_id is not None and next_token_id == eos_token_id:
                final_hidden_state = last_processed_hidden_state
                break

            generated_token_ids.append(next_token_id)
            current_input_ids = next_token.unsqueeze(-1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        device=current_attention_mask.device,
                        dtype=current_attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )

            if step_idx == max_new_tokens - 1 and track_final_hidden_state:
                final_outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                final_hidden_state = final_outputs.hidden_states[-1][:, -1, :].detach().float()
                del final_outputs

            del outputs

    tokenizer.padding_side = original_padding_side

    completion = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return completion, final_hidden_state


def get_final_token_hidden_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
) -> torch.Tensor:
    ensure_pad_token(tokenizer)
    encodings = tokenize_messages(tokenizer, messages)
    encodings = {key: value.to(model.device) for key, value in encodings.items()}
    sequence_length = int(encodings["attention_mask"][0].sum().item())

    with torch.inference_mode():
        outputs = model(
            **encodings,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_state = outputs.hidden_states[-1][0, sequence_length - 1, :].detach().float()
    del outputs
    return hidden_state


def compute_probe_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if PROBE_LOSS == "mse":
        return F.mse_loss(prediction, target)
    if PROBE_LOSS == "cosine":
        return 1.0 - F.cosine_similarity(prediction, target).mean()
    raise ValueError(f"Unsupported probe loss: {PROBE_LOSS}")


def make_student_prompt(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    return build_student_messages(example, completion="")[:1]


def make_teacher_prompt(
    example: Dict[str, Any],
    student_attempt: str,
    feedback_text: str,
) -> List[Dict[str, Any]]:
    return build_teacher_messages(
        example,
        student_attempt=student_attempt,
        feedback_text=feedback_text,
        completion="",
    )[:1]


def teacher_mode_step(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probe: torch.nn.Linear,
    optimizer: torch.optim.Optimizer,
    example: Dict[str, Any],
    example_index: int,
    step_idx: int,
) -> Dict[str, Any]:
    student_attempt, _ = autoregressive_generate(
        model,
        tokenizer,
        make_student_prompt(example),
        temperature=STUDENT_TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    feedback = get_environment_feedback(student_attempt, example)

    teacher_attempt, h_pi = autoregressive_generate(
        model,
        tokenizer,
        make_teacher_prompt(example, student_attempt, feedback.feedback_text),
        temperature=TEACHER_TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        track_final_hidden_state=True,
    )
    if h_pi is None:
        raise RuntimeError("Teacher mode expected a final hidden state from generation but did not receive one.")

    h = get_final_token_hidden_state(
        model,
        tokenizer,
        build_student_messages(example, teacher_attempt),
    )
    h_o = get_final_token_hidden_state(
        model,
        tokenizer,
        build_observation_messages(example, student_attempt, feedback.feedback_text),
    )

    delta = h_pi - h
    prediction = probe(delta.unsqueeze(0))
    target = h_o.unsqueeze(0).to(prediction.device)

    loss = compute_probe_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    metrics = {
        "step": step_idx,
        "example_index": example_index,
        "mode": "teacher",
        "question_title": example["question_title"],
        "student_attempt": student_attempt,
        "teacher_attempt": teacher_attempt,
        "feedback_text": feedback.feedback_text,
        "feedback_success": bool(feedback.success),
        "loss": float(loss.item()),
        "delta_norm": float(delta.norm().item()),
        "target_norm": float(h_o.norm().item()),
        "prediction_norm": float(prediction.detach()[0].norm().item()),
    }

    del h_pi, h, h_o, delta, prediction, target, loss
    gc.collect()
    maybe_empty_cuda_cache()
    return metrics


def student_mode_step(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probe: torch.nn.Linear,
    optimizer: torch.optim.Optimizer,
    example: Dict[str, Any],
    example_index: int,
    step_idx: int,
) -> Dict[str, Any]:
    student_attempt, _ = autoregressive_generate(
        model,
        tokenizer,
        make_student_prompt(example),
        temperature=STUDENT_TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    feedback = get_environment_feedback(student_attempt, example)

    h_pi = get_final_token_hidden_state(
        model,
        tokenizer,
        build_teacher_messages(
            example,
            student_attempt=student_attempt,
            feedback_text=feedback.feedback_text,
            completion=student_attempt,
        ),
    )
    h = get_final_token_hidden_state(
        model,
        tokenizer,
        build_student_messages(example, student_attempt),
    )
    h_o = get_final_token_hidden_state(
        model,
        tokenizer,
        build_observation_messages(example, student_attempt, feedback.feedback_text),
    )

    delta = h_pi - h
    prediction = probe(delta.unsqueeze(0))
    target = h_o.unsqueeze(0).to(prediction.device)

    loss = compute_probe_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    metrics = {
        "step": step_idx,
        "example_index": example_index,
        "mode": "student",
        "question_title": example["question_title"],
        "student_attempt": student_attempt,
        "feedback_text": feedback.feedback_text,
        "feedback_success": bool(feedback.success),
        "loss": float(loss.item()),
        "delta_norm": float(delta.norm().item()),
        "target_norm": float(h_o.norm().item()),
        "prediction_norm": float(prediction.detach()[0].norm().item()),
    }

    del h_pi, h, h_o, delta, prediction, target, loss
    gc.collect()
    maybe_empty_cuda_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["student", "teacher"])
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    args = parser.parse_args()

    configure_logging()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    logging.info("Loading model %s on %s with dtype %s", args.model_name, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ensure_pad_token(tokenizer)
    model.eval()
    model.requires_grad_(False)

    dataset = LiveCodeBenchDataset(subset_size=args.subset_size)
    if len(dataset) == 0:
        raise RuntimeError("LiveCodeBenchDataset is empty.")

    hidden_size = int(model.config.hidden_size)
    probe = torch.nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=PROBE_LEARNING_RATE,
        weight_decay=PROBE_WEIGHT_DECAY,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    step_metrics: List[Dict[str, Any]] = []

    for step_idx in range(NUM_PROBE_STEPS):
        example_index = (args.start_index + step_idx) % len(dataset)
        example = dataset[example_index]
        logging.info(
            "Running %s probe step %d/%d on example %d: %s",
            args.mode,
            step_idx + 1,
            NUM_PROBE_STEPS,
            example_index,
            example["question_title"],
        )

        if args.mode == "teacher":
            metrics = teacher_mode_step(
                model=model,
                tokenizer=tokenizer,
                probe=probe,
                optimizer=optimizer,
                example=example,
                example_index=example_index,
                step_idx=step_idx,
            )
        else:
            metrics = student_mode_step(
                model=model,
                tokenizer=tokenizer,
                probe=probe,
                optimizer=optimizer,
                example=example,
                example_index=example_index,
                step_idx=step_idx,
            )

        step_metrics.append(metrics)
        logging.info(
            "Completed step %d with loss %.6f and feedback_success=%s",
            step_idx,
            metrics["loss"],
            metrics["feedback_success"],
        )

    average_loss = sum(step["loss"] for step in step_metrics) / len(step_metrics)
    success_rate = sum(1 for step in step_metrics if step["feedback_success"]) / len(step_metrics)
    probe_weight_norm = float(probe.weight.detach().norm().item())
    probe_bias_norm = float(probe.bias.detach().norm().item())

    model_stub = args.model_name.replace("/", "_")
    output_stem = f"{args.mode}_{model_stub}_start{args.start_index}"
    summary_path = Path(args.output_dir) / f"{output_stem}.json"
    probe_path = Path(args.output_dir) / f"{output_stem}.pt"

    torch.save(probe.state_dict(), probe_path)

    output = {
        "mode": args.mode,
        "model_name": args.model_name,
        "seed": args.seed,
        "start_index": args.start_index,
        "subset_size": args.subset_size,
        "device": device,
        "dtype": str(dtype),
        "num_probe_steps": NUM_PROBE_STEPS,
        "student_temperature": STUDENT_TEMPERATURE,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "probe_learning_rate": PROBE_LEARNING_RATE,
        "probe_weight_decay": PROBE_WEIGHT_DECAY,
        "probe_loss": PROBE_LOSS,
        "summary": {
            "average_loss": average_loss,
            "success_rate": success_rate,
            "probe_weight_norm": probe_weight_norm,
            "probe_bias_norm": probe_bias_norm,
            "probe_state_dict_path": str(probe_path),
        },
        "steps": step_metrics,
    }

    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)

    logging.info("Saved summary to %s", summary_path)
    logging.info("Saved probe weights to %s", probe_path)


if __name__ == "__main__":
    main()
