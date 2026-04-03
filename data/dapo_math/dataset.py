import os
import json
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from dotenv import load_dotenv


load_dotenv()


def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


class DAPOMathDataset:
    def __init__(self, split: str = "train", num_examples: int = 128):
        self.train_range = range(0, 10_000)
        self.test_range = range(10_000, 10_500)
        
        self.dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        if split == "train":
            self.dataset = self.dataset.select(self.train_range).select(range(num_examples))
        elif split == "test":
            self.dataset = self.dataset.select(self.test_range).select(range(num_examples))
        else:
            raise ValueError("Invalid split. Must be 'train' or 'test.")


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]

    def select(self, indices: List[int]) -> Dataset:
        return self.dataset.select(indices)