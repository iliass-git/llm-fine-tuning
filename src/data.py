import os
import json
from datasets import Dataset

def load_dataset(dataset_path, text_field, use_subset=False, subset_size=1000):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f if line.strip()]
    if text_field not in data_list[0]:
        raise KeyError(f"Field '{text_field}' not found in dataset!")
    dataset = Dataset.from_list(data_list)
    if use_subset:
        dataset = dataset.select(range(min(subset_size, len(dataset))))
    return dataset
