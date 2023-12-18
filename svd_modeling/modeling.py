"""
replace the linear module with the svd module
"""
import sys
from typing import List, Dict

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


def replace_linear_with_svd(
        base_model,
        compress_ratio: float,
        target_module: List[str],
        importance_dict: Dict[str, torch.Tensor] = None,
        memory_efficient: bool = True,
        print_info: bool = False,
):
    """
    replace the linear module with the svd module
    """
    info = []

    def _set_module(model, name, new_module):
        layer = name.split(".")[1]
        cur_mod = model.layers[int(layer)]
        blocks = name.split(".")[2:-1]
        for block in blocks:
            cur_mod = getattr(cur_mod, block)
        setattr(cur_mod, name.split(".")[-1], new_module)

    modules_to_substitute = []
    for name, module in base_model.named_modules():
        if any([target in name for target in target_module]):
            modules_to_substitute.append((name, module))

    for name, module in tqdm(modules_to_substitute, desc="Replacing linear with SVD"):
        A = module.weight.data.T
        num_ranks = calc_rank(A, compress_ratio)

        if importance_dict is not None and name in importance_dict:
            if isinstance(importance_dict[name], torch.Tensor):
                # fisher weighted SVD
                W = importance_dict[name].T
            else:
                # lora weighted SVD
                ...
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

        new_module = SVDLinear(L1, L2, bias=False, svd_init=True)
        if A.dtype == torch.float16:
            new_module = new_module.half()
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
    Do SVD approximation on the weight of linear module
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
        if A.dtype == torch.float16:
            new_A = new_A.half()
        module.weight.data = new_A.T

        if print_info:
            info.append(f"{name}: {A.shape} -> {L1.shape} @ {L2.shape}, error: {calc_error(A, L1, L2):.4f}")

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
    model = AutoModel.from_pretrained(model_name, config=config, cache_dir="../.cache").to("cuda:0")

    iportance_dict = {}
    for n, p in model.named_parameters():
        if any([target in n for target in TARGET_MODULES]):
            iportance_dict[n] = torch.randint_like(p, 0, 2).float().to("cuda:0")

    replace_linear_with_svd(model, 0.1, TARGET_MODULES, importance_dict=iportance_dict, print_info=True)
    print(model)
