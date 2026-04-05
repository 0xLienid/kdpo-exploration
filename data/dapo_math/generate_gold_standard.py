import os
from openai import OpenAI
from datasets import Dataset
from data.dapo_math import DAPOMathDataset, verify_answer
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    dataset = DAPOMathDataset(split="test", num_examples=32)

    dataset_with_gold_standard = []
    for example in dataset:
        attempts = 0

        while attempts < 10:
            response = client.responses.create(
                model="gpt-5.4",
                input=example["prompt"],
            )
            is_correct = verify_answer(response.output_text, example["reward_model"])

            if is_correct:
                break

            attempts += 1

        dataset_with_gold_standard.append({
            **example,
            "gold_standard": response.output_text,
        })

    dataset_with_gold_standard = Dataset.from_list(dataset_with_gold_standard)
    dataset_with_gold_standard.push_to_hub("mfirth/dapo-math-gold-standard", token=os.getenv("HF_TOKEN"))