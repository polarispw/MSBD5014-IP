import torch
from tqdm import tqdm
from datasets import Dataset


def preprocess_for_casuallm(dataset, tokenizer, seq_len, field_name) -> Dataset:
    print("Tokenizing...")
    ids = tokenizer("\n\n".join(dataset[field_name]), return_tensors='pt').input_ids[0]
    input_samples = []
    n_samples = ids.numel() // seq_len

    for i in range(n_samples):
        batch = ids[(i * seq_len): ((i + 1) * seq_len)]
        input_samples.append(batch)

    input_dict = {"input_ids": input_samples, "labels": input_samples}
    input_dict = Dataset.from_dict(input_dict)
    return input_dict
