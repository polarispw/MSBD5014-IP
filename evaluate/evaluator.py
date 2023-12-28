import torch
from datasets import load_dataset, Dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import preprocess_for_causallm


def evaluate_ppl(model, tokenizer, dataset_name, dataset_cache_dir, seq_len=2048):
    if dataset_name == "wikitext":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=dataset_cache_dir, split='test')
        field_name = 'text'
    else:
        raise NotImplementedError

    dataset = preprocess_for_causallm(dataset, tokenizer, seq_len, field_name)
    ppl = eval_ppl(model, dataset)
    return ppl


# the above to be deprecated
def eval_ppl(model, dataset: Dataset) -> float:
    dataset = dataset.with_format("torch", device=model.device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    nlls = []
    for batch in tqdm(dataloader, desc="Evaluating..."):
        with torch.no_grad():
            loss = model(**batch).loss
            nlls.append(loss)

    pll = torch.exp(torch.stack(nlls).mean())
    return pll.item()
