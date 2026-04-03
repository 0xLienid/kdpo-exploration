import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from validators.validator import Validator
from data.dapo_math import DAPOMathDataset, verify_answer


class DAPOMathValidator(Validator):
    def __init__(self):
        super().__init__("dapo")

        self.dataset = DAPOMathDataset(split="test", num_examples=64)

    def compute_local_stats(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 2048,
        max_seq_length: int = 2048,
        process_index: int = 0,
        num_processes: int = 1
    ) -> dict[str, float]:
        if process_index == 0:
            print("Validating DAPO Math...")

        model.eval()
        gc_was_enabled = model.is_gradient_checkpointing
        if gc_was_enabled:
            model.gradient_checkpointing_disable()

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        correct = 0
        local_indices = list(range(process_index, len(self.dataset), num_processes))
        total_batches = (len(local_indices) + batch_size - 1) // batch_size

        for batch_start in range(0, len(local_indices), batch_size):
            if process_index == 0:
                print(f"Processing batch{(batch_start // batch_size) + 1} of {total_batches}...")

            batch_end = min(batch_start + batch_size, len(local_indices))
            batch_indices = local_indices[batch_start:batch_end]
            batch_data = self.dataset.select(batch_indices)

            batch_prompts = []
            for example in batch_data:
                prompt = tokenizer.apply_chat_template(example["prompt"], add_generation_prompt=True, tokenize=False)
                batch_prompts.append(prompt)

            batch_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                padding_side="left"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    tokenizer=tokenizer,
                    do_sample=False
                )

            completions = tokenizer.batch_decode(outputs[:, batch_inputs.input_ids.shape[-1]:], skip_special_tokens=True)

            for j, completion in enumerate(completions):
                ground_truth_dict = batch_data[j]["reward_model"]
                is_correct = verify_answer(completion, ground_truth_dict)

                if is_correct:
                    correct += 1

        if gc_was_enabled:
            model.gradient_checkpointing_enable()

        tokenizer.padding_side = original_padding_side

        return {
            "correct": float(correct),
            "total": float(len(local_indices))
        }

    def compute_score(self, stats: dict[str, float]) -> float:
        total = stats["total"]
        return stats["correct"] / total if total > 0 else 0.0