import torch
from tqdm import tqdm
from datasets import Dataset


def preprocess_for_causallm(dataset, tokenizer, seq_len, field_name, show_progress=False) -> Dataset:
    input_samples = []
    if not show_progress:
        print("Tokenizing...")
        ids = tokenizer("\n\n".join(dataset[field_name]), return_tensors='pt').input_ids[0]
        n_samples = ids.numel() // seq_len
        print(f"Splitting into {n_samples} samples...")
    else:
        input_samples = []
        context = "\n\n".join(dataset[field_name])
        context_cache = []
        for i in tqdm(range(0, len(context), seq_len), desc="Tokenizing..."):
            context_cache.append(tokenizer(context[i:i + seq_len], return_tensors='pt').input_ids[0])
        ids = torch.cat(context_cache)
        n_samples = ids.numel() // seq_len
        print(f"Splitting into {n_samples} samples...")

    for i in range(n_samples):
        batch = ids[(i * seq_len): ((i + 1) * seq_len)]
        input_samples.append(batch)

    input_dict = {"input_ids": input_samples, "labels": input_samples}
    input_dict = Dataset.from_dict(input_dict)
    return input_dict
