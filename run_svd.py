"""
Get Low-rank Decomposition of a model.
"""
import torch

torch.manual_seed(42)

linear1 = torch.nn.Linear(4096, 256)
linear2 = torch.nn.Linear(256, 4096)

lora_a = torch.nn.Linear(4096, 32)
lora_b = torch.nn.Linear(32, 4096)

comb_linear1 = torch.nn.Linear(4096, 256)
comb_linear2 = torch.nn.Linear(256, 4096)

gap = 256 - 32
zero_pad_lora_a = torch.cat([lora_a.weight.data, torch.zeros((gap, 4096))], dim=0)
zero_pad_lora_b = torch.cat([lora_b.weight.data, torch.zeros((4096, gap))], dim=1)
comb_linear1.weight.data = linear1.weight.data + zero_pad_lora_a
comb_linear2.weight.data = linear2.weight.data + zero_pad_lora_b

input_tensor = torch.randn(1, 4096)

output1 = linear2(linear1(input_tensor)) + lora_b(lora_a(input_tensor))
output2 = comb_linear2(comb_linear1(input_tensor))

error = torch.norm(output1 - output2)
print(error.item())
