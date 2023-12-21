import torch
from datasets import load_dataset, Dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import preprocess_for_casuallm


class ManualPPLMetric:
    def __init__(self):
        self.nlls = []
        self.loss_fct = nn.CrossEntropyLoss()

    def add_nll(self, lm_logits, batch, seqlen):
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * seqlen
        self.nlls.append(neg_log_likelihood)

    def calculate_ppl(self, nsamples, seqlen):
        ppl = torch.exp(torch.stack(self.nlls).sum() / (nsamples * seqlen))
        return ppl.item()


def evaluate_ppl(model, tokenizer, dataset_name, dataset_cache_dir, seq_len=2048):
    if dataset_name == "wikitext":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=dataset_cache_dir, split='test')
        field_name = 'text'
    else:
        raise NotImplementedError

    dataset = preprocess_for_casuallm(dataset, tokenizer, seq_len, field_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    metric = ManualPPLMetric()

    for batch in tqdm(dataloader, desc="Evaluating..."):
        batch = batch.to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        metric.add_nll(lm_logits, batch, seq_len)
    ppl = metric.calculate_ppl(len(dataset), seq_len)
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
