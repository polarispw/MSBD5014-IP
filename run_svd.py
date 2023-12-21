"""
Get Low-rank Decomposition of a model.
"""
import torch

a = torch.load('.cache/llama1b_fisher.pt')

for k, v in a.items():
    print(k, v.shape)
