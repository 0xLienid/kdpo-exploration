import torch
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.livecodebench import LiveCodeBenchDataset, get_environment_feedback, rollout


def log_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available")
        return

    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3

    print(
        f"[{label}] "
        f"allocated={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB, "
        f"peak_allocated={peak_allocated:.2f} GB, "
        f"peak_reserved={peak_reserved:.2f} GB"
    )


def build_teacher_prompt(question: str, student_attempt: str, feedback: str) -> List[Dict[str, Any]]:
    return [
        {"role": "user",
            "content": f"## Question\n{question}\n\n## Previous Attempt\n{student_attempt}\n\n## Feedback (from environment) for the previous attempt\n{feedback}\n\nCorrectly solve the original question. If the previous attempt is cut off, make sure to be more efficient and concise in your next attempt. You may refer to the previous attempt during the thinking phase of your next attempt, but do not refer to it at all in your final answer. Keep your thinking concise, you have 2048 tokens to work with, including the final answer."}
    ]


def teacher_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Dict[str, Any],
    rollouts: List[Any],
    feedbacks: List[Any],
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
) -> List[Dict[str, Any]]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        enable_thinking=True
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


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

    dataset = LiveCodeBenchDataset()

    example = dataset[0]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_cuda_memory("before student rollout")
    rollouts = rollout(
        model,
        tokenizer,
        example,
        num_rollouts=1,
        temperature=1.0,
        max_new_tokens=4096,
    )
    log_cuda_memory("after student rollout")
    feedbacks = [get_environment_feedback(rollout["completion"], example) for rollout in rollouts]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_cuda_memory("before teacher rollout")
    teacher_rollouts = teacher_rollout(
        model,
        tokenizer,
        example,
        rollouts,
        feedbacks,
        temperature=1.0,
        max_new_tokens=32768,
    )
    log_cuda_memory("after teacher rollout")
    teacher_feedbacks = [get_environment_feedback(teacher_rollout["completion"], example) for teacher_rollout in teacher_rollouts]

    print(rollouts[0]["completion"])
    print(tokenizer.encode(rollouts[0]["completion"], return_tensors="pt").shape)
    print(feedbacks)

    print(teacher_rollouts[0]["completion"])
    print(tokenizer.encode(teacher_rollouts[0]["completion"], return_tensors="pt").shape)
    print(teacher_feedbacks)