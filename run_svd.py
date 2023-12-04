"""
Get Low-rank Decomposition of a model.
"""
import torch

torch.manual_seed(42)

linear = torch.nn.Linear(2048, 5504, bias=False)
w = linear.weight.data.T
print(f"w: {w.shape}")

num_ranks = 100
# U, S, VT = torch.linalg.svd(w, full_matrices=False)
U, S, V = torch.svd_lowrank(w, num_ranks)
VT = V.mH
print(f"U: {U.shape}, S: {S.shape}, VT: {VT.shape}")

S_sqrt = torch.sqrt(S)
L1 = U * S_sqrt.unsqueeze(dim=0)
L2 = VT * S_sqrt.unsqueeze(dim=1)
L1k = L1[:, :num_ranks]
L2k = L2[:num_ranks, :]
print(f"L1k: {L1k.shape}, L2k: {L2k.shape}")

linear1 = torch.nn.Linear(2048, num_ranks, bias=False)
linear2 = torch.nn.Linear(num_ranks, 5504, bias=False)

linear1.weight.data = L1k.T
linear2.weight.data = L2k.T

input_ids = torch.randn(1, 2048)

output1 = linear(input_ids)
output2 = linear1(input_ids)
output2 = linear2(output2)

error = torch.linalg.norm(output1 - output2, ord="fro")
print(f"error: {error}")
