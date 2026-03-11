from dataclasses import dataclass


@dataclass
class ValidatorRunConfig:
    batch_size: int = 4
    max_new_tokens: int = 2048
    max_seq_length: int = 2048
