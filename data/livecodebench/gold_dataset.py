import json
import zlib
import pickle
import base64
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset


def _decode_private_test_cases(private_test_cases_raw: str) -> list[dict[str, Any]]:
    try:
        private_test_cases = pickle.loads(
            zlib.decompress(base64.b64decode(private_test_cases_raw.encode("utf-8")))
        )
    except Exception as e:
        print(f"Error loading private tests: {e}")
        private_test_cases = []

    if isinstance(private_test_cases, str):
        return json.loads(private_test_cases)
    return private_test_cases


class LiveCodeBenchGoldStandardDataset:
    REPO_ID = "mfirth/livecodebench-gold-standard"

    def __init__(self, subset_size: Optional[int] = None):
        self.subset_size = subset_size

        self._dataset = load_dataset(self.REPO_ID, split="train")
        self._dataset = self._dataset.map(self._load_tests)

        if subset_size:
            self._dataset = self._dataset.select(
                range(min(subset_size, len(self._dataset))))

    def _load_tests(self, example: Dict[str, Any]) -> Dict[str, Any]:
        public_tests = example["public_test_cases"]
        if isinstance(public_tests, str):
            public_tests = json.loads(public_tests)

        private_tests_raw = example["private_test_cases"]
        private_tests = private_tests_raw
        if isinstance(private_tests_raw, str):
            private_tests = _decode_private_test_cases(private_tests_raw)

        example["public_test_cases"] = public_tests
        example["private_test_cases"] = private_tests
        return example

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._dataset[idx]

    def select(self, indices: List[int]) -> Dataset:
        return self._dataset.select(indices)
