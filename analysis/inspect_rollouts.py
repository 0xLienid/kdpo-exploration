import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.livecodebench import LiveCodeBenchDataset, get_environment_feedback, rollout

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")

    dataset = LiveCodeBenchDataset()

    example = dataset[0]
    rollout = rollout(
        model,
        tokenizer,
        example,
        num_rollouts=1,
        temperature=1.0,
        max_new_tokens=2048,
    )
    feedback = get_environment_feedback(rollout["completion"], example)

    print(rollout)
    print(feedback)