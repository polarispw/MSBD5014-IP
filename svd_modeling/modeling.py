"""
replace the linear module with the svd module
"""
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

    # the length is uncertain, so the bar doesn't show
    for name, module in tqdm(base_model.named_modules(), desc="Replacing linear with SVD"):
        if any([target in name for target in target_module]):

            num_ranks = calc_rank(module.weight.data, compress_ratio)

            if importance_dict is not None and name in importance_dict:
                L1, L2 = weighted_svd_decomposition(
                    module.weight.data.T,
                    importance_dict[name].T,
                    heuristic="two-sided",
                    num_ranks=num_ranks,
                    randomized=True,
                    num_oversampling=10,
                    normalize=False,
                    reduce_before_sqrt=True
                )
            else:
                L1, L2 = svd_decomposition(
                    module.weight.data.T,
                    randomized=True,
                    num_ranks=num_ranks,
                    num_oversampling=10,
                )

            if print_info:
                info.append(f"{name}: {module.weight.data.T.shape} -> {L1.shape} * {L2.shape}, "
                            f"error: {calc_error(module.weight.data.T, L1, L2):.4f}")

            new_module = SVDLinear(L1, L2, bias=False, svd_init=True)
            _set_module(base_model, name, new_module)

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

    ipt_dic = {}
    for n, p in model.named_parameters():
        if any([target in n for target in TARGET_MODULES]):
            ipt_dic[n] = torch.randint_like(p, 0, 2).float().to("cuda:0")

    replace_linear_with_svd(model, 0.1, TARGET_MODULES, importance_dict=ipt_dic, print_info=True)
    print(model)
