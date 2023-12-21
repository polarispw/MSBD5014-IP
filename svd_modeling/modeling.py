"""
build the SVD model from the original model
"""
import sys
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .decompose import (
    calc_rank,
    calc_error,
    svd_decomposition,
    weighted_svd_decomposition
)


class SVDLinear(nn.Module):
    def __init__(self, L1, L2, bias=False, svd_init=True):
        super(SVDLinear, self).__init__()
        self.low_rank = L1.shape[1]
        self.linear1 = nn.Linear(L1.shape[0], self.low_rank, bias=bias)
        self.linear2 = nn.Linear(self.low_rank, L2.shape[1], bias=bias)

        if svd_init:
            self.linear1.weight.data = L1.T
            self.linear2.weight.data = L2.T

    def forward(self, input):
        output = self.linear1(input)
        output = self.linear2(output)
        return output


class LoRASVDLinear(SVDLinear):
    """
    further optimization is necessary to integrate the LoRA module into the SVD module
    """
    def __init__(self, L1, L2, la, lb, bias=False, svd_init=True):
        super(LoRASVDLinear, self).__init__(L1, L2, bias=bias, svd_init=svd_init)
        self.lora_rank = la.shape[1]
        self.lora_A = nn.Linear(la.shape[0], self.lora_rank, bias=bias)
        self.lora_B = nn.Linear(self.lora_rank, lb.shape[1], bias=bias)

        if svd_init:
            self.lora_A.weight.data = la
            self.lora_B.weight.data = lb

    def forward(self, input):
        stem_output = self.linear1(input)
        stem_output = self.linear2(stem_output)
        lora_output = self.lora_A(input)
        lora_output = self.lora_B(lora_output)
        return stem_output + lora_output

    def freeze_stem(self):
        self.linear1.weight.requires_grad = False
        self.linear2.weight.requires_grad = False


# get target module in LLaMA, it may differ in different models
def _set_module(model, name, new_module):
    layer = name.split(".")[1]
    cur_mod = model.layers[int(layer)]
    blocks = name.split(".")[2:-1]
    for block in blocks:
        cur_mod = getattr(cur_mod, block)
    setattr(cur_mod, name.split(".")[-1], new_module)


def linear_to_svdlinear(
        base_model,
        compress_ratio: float,
        target_module: List[str],
        importance_dict: [Dict[str, torch.Tensor], Dict[str, Tuple]] = None,
        memory_efficient: bool = True,
        print_info: bool = False,
):
    """
    replace the linear module with the svd linear module
    A -> (X @ L1) @ L2
    """
    info = []
    modules_to_substitute = []
    for name, module in base_model.named_modules():
        if any([target in name for target in target_module]):
            modules_to_substitute.append((name, module))

    for name, module in tqdm(modules_to_substitute, desc="Replacing linear with SVD-Linear"):
        A = module.weight.data.T
        num_ranks = calc_rank(A, compress_ratio)
        la, lb = None, None

        if importance_dict is not None and name in importance_dict:
            if isinstance(importance_dict[name], Tuple):
                # lora weighted SVD
                la, lb = importance_dict[name]
                W = la @ lb
            else:
                # fisher weighted SVD
                W = importance_dict[name].T

            L1, L2 = weighted_svd_decomposition(
                A,
                W,
                heuristic="two-sided",
                num_ranks=num_ranks,
                randomized=True,
                num_oversampling=10,
                normalize=False,
                reduce_before_sqrt=True
            )
        else:
            L1, L2 = svd_decomposition(
                A,
                randomized=True,
                num_ranks=num_ranks,
                num_oversampling=10,
            )

        if la is not None and lb is not None:
            new_module = LoRASVDLinear(L1, L2, la, lb, bias=False, svd_init=True)
        elif la is None and lb is None:
            new_module = SVDLinear(L1, L2, bias=False, svd_init=True)
        else:
            raise NotImplementedError

        _set_module(base_model, name, new_module)
        # clear the original weight to save memory
        if memory_efficient:
            module.weight.data = torch.tensor([]).to(module.weight.device)

        if print_info:
            info.append(f"{name}: {A.shape} -> {L1.shape} * {L2.shape}, error: {calc_error(A, L1, L2):.4f}")

    if print_info:
        print("\n".join(info))

    return base_model


def svd_approximation(
        base_model,
        compress_ratio: float,
        target_module: List[str],
        importance_dict: Dict[str, torch.Tensor] = None,
        print_info: bool = False,
):
    """
    Do SVD approximation on the weight of linear module then reconstruct it
    """
    info = []
    modules_to_process = []
    for name, module in base_model.named_modules():
        if any([target in name for target in target_module]):
            modules_to_process.append((name, module))

    for name, module in tqdm(modules_to_process, desc="Conducting SVD approximation"):
        A = module.weight.data.T
        num_ranks = calc_rank(A, compress_ratio)

        if importance_dict is not None and name in importance_dict:
            W = importance_dict[name].T
            L1, L2 = weighted_svd_decomposition(
                A,
                W,
                heuristic="two-sided",
                num_ranks=num_ranks,
                randomized=True,
                num_oversampling=10,
                normalize=False,
                reduce_before_sqrt=True
            )
        else:
            L1, L2 = svd_decomposition(
                A,
                randomized=True,
                num_ranks=num_ranks,
                num_oversampling=10,
            )
        new_A = L1 @ L2
        module.weight.data = new_A.T

        if print_info:
            info.append(f"{name}: {A.shape} ~ {L1.shape} @ {L2.shape}, error: {calc_error(A, L1, L2):.4f}")

    if print_info:
        print("\n".join(info))

    return base_model


if __name__ == "__main__":
    # test
    from transformers import AutoModel, AutoConfig
    from svd_modeling import TARGET_MODULES

    model_name = "meta-llama/Llama-2-7b-hf"
    config = AutoConfig.from_pretrained(model_name)
    config.update({"num_hidden_layers": 2})
    model = AutoModel.from_pretrained(model_name, config=config, cache_dir="../.cache").half().to("cuda:0")

    importance_dict = {}
    for n, p in model.named_parameters():
        if any([target in n for target in TARGET_MODULES]):
            pt = p.T
            la = torch.randint_like(pt[:, :8], 0, 2).float().to("cuda:0")
            lb = torch.randint_like(pt[:8, :], 0, 2).float().to("cuda:0")
            importance_dict[n[:-7]] = (la, lb)
            # importance_dict[n[:-7]] = torch.ones_like(p).float().to("cuda:0")

    linear_to_svdlinear(model, 0.1, TARGET_MODULES, importance_dict=importance_dict, print_info=True)
    print(model)
    print(model.layers[0].mlp.up_proj.linear1.weight.data.dtype)
