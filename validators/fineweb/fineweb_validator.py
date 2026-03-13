import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from validators.validator import Validator


class FineWebValidator(Validator):
    def __init__(self):
        super().__init__("fineweb")

        self.test_dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    def compute_local_stats(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 0,  # unused for perplexity, kept for interface consistency
        max_seq_length: int = 1024,
        num_samples: int = 16,
        process_index: int = 0,
        num_processes: int = 1,
    ) -> dict[str, float]:
        if process_index == 0:
            print("Validating FineWeb...")

        model.eval()

        original_padding_side = tokenizer.padding_side
        original_pad_token = tokenizer.pad_token
        original_pad_token_id = tokenizer.pad_token_id

        try:
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token

            dataset_iter = iter(self.test_dataset)

            total_nll = 0.0
            total_token_count = 0

            batch = []
            for sample_index in range(num_samples):
                try:
                    sample = next(dataset_iter)
                except StopIteration:
                    break

                if sample_index % num_processes != process_index:
                    continue

                batch.append(sample["text"])
                if len(batch) < batch_size:
                    continue

                batch_inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    padding_side="left",
                ).to(model.device)

                labels = batch_inputs["input_ids"].clone()
                if "attention_mask" in batch_inputs:
                    labels[batch_inputs["attention_mask"] == 0] = -100

                with torch.no_grad():
                    outputs = model(**batch_inputs)
                    logits = outputs.logits.float()

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    loss_sum = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )

                    valid_tokens = (shift_labels != -100).sum().item()
                    if valid_tokens > 0:
                        total_nll += loss_sum.item()
                        total_token_count += valid_tokens

                batch = []

            if batch:
                batch_inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    padding_side="left",
                ).to(model.device)

                labels = batch_inputs["input_ids"].clone()
                if "attention_mask" in batch_inputs:
                    labels[batch_inputs["attention_mask"] == 0] = -100

                with torch.no_grad():
                    outputs = model(**batch_inputs)
                    logits = outputs.logits.float()

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    loss_sum = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )

                    valid_tokens = (shift_labels != -100).sum().item()
                    if valid_tokens > 0:
                        total_nll += loss_sum.item()
                        total_token_count += valid_tokens

            return {
                "total_nll": total_nll,
                "total_token_count": float(total_token_count),
            }
        finally:
            tokenizer.padding_side = original_padding_side
            tokenizer.pad_token = original_pad_token
            tokenizer.pad_token_id = original_pad_token_id

    def compute_score(self, stats: dict[str, float]) -> float:
        total_token_count = stats["total_token_count"]
        if total_token_count == 0:
            return float("inf")

        average_nll = stats["total_nll"] / total_token_count
        return math.exp(average_nll)
