"""
Get Low-rank Decomposition of a model.
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# eval llama 1.3b on wikitext-2
from evaluate import preprocess_for_casuallm

model = AutoModelForCausalLM.from_pretrained(
    "princeton-nlp/Sheared-LLaMA-1.3B",
    cache_dir=".cache",
    torch_dtype=torch.float16
).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B", cache_dir=".cache")


dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=".data", split='test')
dataset = dataset.select(range(200))
dataset = preprocess_for_casuallm(dataset, tokenizer, 2048, "text")
dataset = dataset.with_format("torch", device=model.device)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

nlls = []
for batch in tqdm(dataloader):
    with torch.no_grad():
        loss = model(**batch).loss
        nlls.append(loss)

pll = torch.exp(torch.stack(nlls).mean())
print(pll.item())
