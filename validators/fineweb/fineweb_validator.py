import math
from itertools import islice
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from validators.validator import Validator


class FineWebValidator(Validator):
    def __init__(self):
        super().__init__("fineweb")
        self.test_dataset: Optional[Dataset] = None
        self.cached_num_samples = 0

    def _ensure_dataset(
        self,
        accelerator,
        num_samples: int,
    ) -> None:
        if self.test_dataset is not None and self.cached_num_samples >= num_samples:
            return

        sampled_rows = [None]
        if accelerator is None or accelerator.is_main_process:
            streaming_dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                "sample-10BT",
                split="train",
                streaming=True,
            )
            sampled_rows[0] = [
                {"text": row["text"]}
                for row in islice(streaming_dataset, num_samples)
            ]

        if accelerator is not None and accelerator.num_processes > 1:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized before broadcasting FineWeb samples."
                )
            dist.broadcast_object_list(sampled_rows, src=0)

        self.test_dataset = Dataset.from_list(sampled_rows[0] or [])
        self.cached_num_samples = len(self.test_dataset)

    def validate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 1024,
        max_seq_length: int = 2048,
        accelerator=None,
        **kwargs,
    ) -> float:
        num_samples = kwargs.get("num_samples", 128)
        self._ensure_dataset(accelerator=accelerator, num_samples=num_samples)
        return super().validate(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            max_seq_length=max_seq_length,
            accelerator=accelerator,
            **kwargs,
        )

    def compute_local_stats(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 0,  # unused for perplexity, kept for interface consistency
        max_seq_length: int = 1024,
        num_samples: int = 128,
        process_index: int = 0,
        num_processes: int = 1,
    ) -> dict[str, float]:
        if process_index == 0:
            print("Validating FineWeb...")

        model.eval()
        local_dataset = self.test_dataset.shard(
            num_shards=num_processes,
            index=process_index,
            contiguous=True,
        )

        original_padding_side = tokenizer.padding_side
        original_pad_token = tokenizer.pad_token
        original_pad_token_id = tokenizer.pad_token_id

        try:
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token

            total_nll = 0.0
            total_token_count = 0
            total_examples = min(num_samples, len(local_dataset))
            total_batches = (total_examples + batch_size - 1) // batch_size

            for batch_start in range(0, total_examples, batch_size):
                batch_end = min(batch_start + batch_size, total_examples)
                batch = local_dataset[batch_start:batch_end]["text"]
                if process_index == 0:
                    print(
                        f"Processing batch {(batch_start // batch_size) + 1} of {total_batches}..."
                    )

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
