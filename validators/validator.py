from typing import Dict, Optional

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer


class Validator:
    def __init__(self, name: str):
        self.name = name

    def validate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 1024,
        max_seq_length: int = 2048,
        accelerator: Optional[Accelerator] = None,
        **kwargs,
    ) -> float:
        process_index = accelerator.process_index if accelerator is not None else 0
        num_processes = accelerator.num_processes if accelerator is not None else 1

        local_stats = self.compute_local_stats(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            max_seq_length=max_seq_length,
            process_index=process_index,
            num_processes=num_processes,
            **kwargs,
        )
        if accelerator is None:
            return self.compute_score(local_stats)

        reduced_stats = {}
        for key, value in local_stats.items():
            stat_tensor = torch.tensor(
                value, device=accelerator.device, dtype=torch.float64)
            reduced_stats[key] = accelerator.reduce(
                stat_tensor, reduction="sum").item()
        return self.compute_score(reduced_stats)

    def compute_local_stats(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 1024,
        max_seq_length: int = 2048,
        process_index: int = 0,
        num_processes: int = 1,
        **kwargs,
    ) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement compute_local_stats()")

    def compute_score(self, stats: Dict[str, float]) -> float:
        raise NotImplementedError("Subclasses must implement compute_score()")
