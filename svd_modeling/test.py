from typing import List

import torch
from transformers import AutoModelForCausalLM

from svd_modeling import TARGET_MODULES

lora = torch.load("../.cache/checkpoint-94/adapter_model.bin")


def get_lora_mask(
        model,
        target_modules: List[str],
        dir_list: List,
):
    # init weight dict
    ipt_dict = {}
    for name, param in model.named_parameters():
        if any([target in name for target in target_modules]):
            ipt_dict[name] = []
            print(name)

    # collect lora weights from dir_list
    prefix = "base_model.model."
    for lora_dir in dir_list:
        lora = torch.load(lora_dir)
        for n in lora.keys():
            print(n)
        for name in ipt_dict.keys():
            name1 = ".".join(name.split(".")[:-1]) + ".lora_A.weight"
            name1 = prefix + name1
            name2 = ".".join(name.split(".")[:-1]) + ".lora_B.weight"
            name2 = prefix + name2
            ipt_dict[name].append((lora[name1].T, lora[name2].T))

    return ipt_dict


model = AutoModelForCausalLM.from_pretrained(
    "princeton-nlp/Sheared-LLaMA-1.3B",
    cache_dir="../.cache",
    torch_dtype=torch.float16
)
target_params = TARGET_MODULES
mask = get_lora_mask(model, target_params, ["../.cache/checkpoint-94/adapter_model.bin"])
for k, v in mask.items():
    print(k, v[0][0].shape[0], v[0][1].shape[1])
