from typing import List

import torch

lora = torch.load("../.cache/checkpoint-94/adapter_model.bin")

for k, v in lora.items():
    print(k, v.shape)

def get_lora_mask(
    dir_list: List,
    calc_relative_vary: bool=False,
):
    # load lora weights
    for 