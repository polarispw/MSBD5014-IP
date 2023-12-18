from typing import List

import torch

lora = torch.load("../.cache/checkpoint-94/adapter_model.bin")

for k, v in lora.items():
    print(k, v.shape)
