import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


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
