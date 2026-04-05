import os
import json
import zlib
import pickle
import base64
from typing import List, Dict, Optional, Any
from datasets import Dataset
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv


load_dotenv()


def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def format_question(example: Dict[str, Any]) -> str:
    return f"{example['question_title']}:\n{example['question_content']}"


class LiveCodeBenchDataset:
    FILES = [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ]

    def __init__(self, data_dir: str = "data_raw/livecodebench", subset_size: Optional[int] = None):
        self.data_dir = data_dir
        self.subset_size = subset_size

        self._download_files()
        self._dataset = self._load_dataset()
        self._dataset = self._dataset.map(self._load_tests)

        if subset_size:
            self._dataset = self._dataset.select(
                range(min(subset_size, len(self._dataset))))

    def _download_files(self):
        os.makedirs(self.data_dir, exist_ok=True)

        for file in self.FILES:
            if not os.path.exists(os.path.join(self.data_dir, file)):
                hf_hub_download(
                    repo_id="livecodebench/code_generation_lite",
                    filename=file,
                    repo_type="dataset",
                    local_dir=self.data_dir,
                    token=os.getenv("HF_TOKEN")
                )

    def _load_dataset(self) -> Dataset:
        data = []

        for file in self.FILES:
            filepath = os.path.join(self.data_dir, file)

            with open(filepath, "r") as f:
                for line in f:
                    data.append(json.loads(line))

        return Dataset.from_list(data)

    def _load_tests(self, example: Dict[str, Any]) -> Dict[str, Any]:
        public_tests = json.loads(example["public_test_cases"])

        private_tests_raw = example["private_test_cases"]
        private_tests = []

        try:
            private_tests = pickle.loads(zlib.decompress(
                base64.b64decode(private_tests_raw.encode("utf-8"))))
        except Exception as e:
            print(f"Error loading private tests: {e}")

        example["public_test_cases"] = public_tests
        example["private_test_cases"] = private_tests

        return example

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self._dataset[idx]
        example["private_test_cases"] = json.loads(example["private_test_cases"])
        return example

    def select(self, indices: List[int]) -> Dataset:
        return self._dataset.select(indices)
