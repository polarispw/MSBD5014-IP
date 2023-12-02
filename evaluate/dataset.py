import torch
from torch.utils.data.dataset import Dataset


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def hf_collator(features):
    batch = {"input_ids": torch.stack(features, dim=0)}
    batch["labels"] = batch["input_ids"].clone()
    return batch


def process_data(dataset, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(dataset[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    n_samples = test_ids.numel() // seq_len

    for i in range(n_samples):
        batch = test_ids[(i * seq_len): ((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)
