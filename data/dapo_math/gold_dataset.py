from datasets import Dataset, load_dataset

from data.dapo_math.dataset import DAPOMathDataset


class DAPOMathGoldStandardDataset(DAPOMathDataset):
    REPO_ID = "mfirth/dapo-math-gold-standard"
    TEST_REPO_ID = "BytedTsinghua-SIA/DAPO-Math-17k"

    def __init__(self, split: str = "train", num_examples: int = 128):
        self.train_range = range(0, 32)
        self.test_range = range(10_000, 10_500)

        self.dataset = load_dataset(self.REPO_ID, split="train")
        self.test_dataset = load_dataset(self.TEST_REPO_ID, split="train")
        if split == "train":
            self.dataset = self.dataset.select(self.train_range)
        elif split == "test":
            self.dataset = self.test_dataset.select(self.test_range).select(range(num_examples))
        else:
            raise ValueError("Invalid split. Must be 'train' or 'test.")

    def select(self, indices: list[int]) -> Dataset:
        return self.dataset.select(indices)
