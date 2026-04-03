import re

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from validators.validator import Validator


class AIME2025Validator(Validator):
    def __init__(self):
        super().__init__("aime2025")

        ds_i = load_dataset("opencompass/AIME2025",
                            name="AIME2025-I", split="test")
        ds_ii = load_dataset("opencompass/AIME2025",
                             name="AIME2025-II", split="test")
        self.dataset = concatenate_datasets([ds_i, ds_ii])

        self.NUMBER_RE = re.compile(r"[+-]?\d[\d,]*")
        self.INVALID_ANS = "[invalid]"

    def extract_boxed_answer(self, text: str) -> str | None:
        def find_matching_brace(s: str, start: int) -> int:
            count = 1
            i = start
            while i < len(s) and count > 0:
                if s[i] == "{":
                    count += 1
                elif s[i] == "}":
                    count -= 1
                i += 1
            return i - 1 if count == 0 else -1

        boxed_start = text.rfind("\\boxed{")
        if boxed_start == -1:
            return None
        
        content_start = boxed_start + 7  # len('\\boxed{')
        closing_brace = find_matching_brace(text, content_start)

        if closing_brace == -1:
            return None

        return text[content_start:closing_brace]

    def normalize_answer(self, text: str) -> str:
        candidate = text.strip()
        if not candidate:
            return self.INVALID_ANS

        while candidate:
            updated = candidate
            if updated.startswith("$") and updated.endswith("$") and len(updated) >= 2:
                updated = updated[1:-1].strip()
            elif updated.startswith("\\(") and updated.endswith("\\)") and len(updated) >= 4:
                updated = updated[2:-2].strip()
            elif updated.startswith("{") and updated.endswith("}") and len(updated) >= 2:
                updated = updated[1:-1].strip()

            if updated == candidate:
                break
            candidate = updated

        candidate = re.sub(r"\s+", "", candidate).rstrip(".")
        if not self.NUMBER_RE.fullmatch(candidate):
            return self.INVALID_ANS

        return str(int(candidate.replace(",", "")))

    def extract_answer(self, text: str) -> str:
        boxed_answer = self.extract_boxed_answer(text)
        if boxed_answer is None:
            return self.INVALID_ANS
        return self.normalize_answer(boxed_answer)

    def verify_answer(self, completion: str, reference: str) -> bool:
        completion_answer = self.extract_answer(completion)
        reference_answer = self.normalize_answer(str(reference))

        if completion_answer == self.INVALID_ANS or reference_answer == self.INVALID_ANS:
            return False

        return completion_answer == reference_answer

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
            print("Validating AIME2025...")

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
                print(f"Processing batch {(batch_start // batch_size) + 1} of {total_batches}...")

            batch_end = min(batch_start + batch_size, len(local_indices))
            batch_indices = local_indices[batch_start:batch_end]
            batch_data = self.dataset.select(batch_indices)

            batch_prompts = []
            for example in batch_data:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"Answer the following question, please keep your reasoning concise and put your code in the typical LaTeX \\boxed{{}} format.\n\nQuestion:\n{example['question']}"}],
                    add_generation_prompt=True,
                    tokenize=False
                )
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

            completions = tokenizer.batch_decode(
                    outputs[:, batch_inputs.input_ids.shape[-1]:], skip_special_tokens=True)

            for j, completion in enumerate(completions):
                ground_truth = batch_data[j]["answer"]
                if self.verify_answer(completion, ground_truth):
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
