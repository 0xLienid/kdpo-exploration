import os
import json
import zlib
import pickle
import base64
from openai import OpenAI
from datasets import Dataset
from dotenv import load_dotenv

from data.livecodebench import (
    LiveCodeBenchDataset,
    extract_python_code,
    format_question,
    run_test_cases,
)


load_dotenv()


def encode_private_test_cases(private_test_cases):
    if not isinstance(private_test_cases, str):
        private_test_cases = json.dumps(private_test_cases)

    return base64.b64encode(
        zlib.compress(pickle.dumps(private_test_cases))
    ).decode("utf-8")


if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    dataset = LiveCodeBenchDataset(subset_size=2)

    dataset_with_gold_standard = []
    for example in dataset:
        attempts = 0

        while attempts < 10:
            response = client.responses.create(
                model="gpt-5.4",
                input=(
                    "Answer the following question, please keep your reasoning concise, "
                    "and put your code in a ```python{code}``` block:\n\n"
                    f"{format_question(example)}"
                ),
            )

            code = extract_python_code(response.output_text)
            is_correct, _ = run_test_cases(
                code,
                test_cases=example["public_test_cases"] + example["private_test_cases"],
            )

            if is_correct:
                break

            attempts += 1

        dataset_with_gold_standard.append({
            **example,
            "public_test_cases": json.dumps(example["public_test_cases"]),
            "private_test_cases": encode_private_test_cases(example["private_test_cases"]),
            "gold_standard": response.output_text,
        })

    dataset_with_gold_standard = Dataset.from_list(dataset_with_gold_standard)
    dataset_with_gold_standard.push_to_hub(
        "mfirth/livecodebench-gold-standard",
        token=os.getenv("HF_TOKEN"),
    )
