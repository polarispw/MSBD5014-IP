import os
import json
import torch
from tqdm import tqdm
from torch import nn
from transformers import AutoModel, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map


def profile_llama():
    model_name = "meta-llama/Llama-2-7b-hf"
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=".cache")
    with init_empty_weights():
        model = AutoModel.from_config(model_config)

    device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer"],)
    my_device_map = device_map

    my_device_map["embed_tokens"] = "cpu"
    my_device_map["norm"] = "cpu"

    start = 7   # from 0 to 7
    gpu_layers = range(start * 4, start * 4 + 4)
    for i in range(model_config.num_hidden_layers):
        if i in gpu_layers:
            my_device_map[f"layers.{i}"] = 0
        elif my_device_map[f"layers.{i}"] == 0:
            my_device_map[f"layers.{i}"] = "cpu"
    print(my_device_map)

    # load
    model = AutoModel.from_pretrained(model_name,
                                      cache_dir=".cache",
                                      device_map=my_device_map,
                                      # device_map=device_map,
                                      )

    matrix_dics = {}
    for n, p in model.named_parameters():
        if p.device == torch.device("cuda:0"):
            matrix_dics[n] = p

    # profile matrices
    matrix_profiles = {}
    compress_factor = 0.5
    for n, p in tqdm(matrix_dics.items()):
        size = list(p.shape)
        s90 = []
        if len(size) == 2:
            top_r = int((size[0] * size[1]) / (size[0] + size[1]) * compress_factor)
            u, s, vh = torch.linalg.svd(p)
            reconstruct_p = u[:, :top_r] @ (torch.diag(s[:top_r]) @ vh[:top_r, :])
            p1_err = float(torch.dist(p, reconstruct_p, p=1).detach().cpu() / (size[0] * size[1]))
            relative_err = p1_err / float(torch.dist(p, torch.zeros_like(p), p=1).detach().cpu() / (size[0] * size[1])) * 100
            p2_err = float(torch.dist(p, reconstruct_p, p=2).detach().cpu())
            # u, s, vh = torch.linalg.svd(p, full_matrices=False)
            # err = float(torch.dist(p, u @ (torch.diag(s) @ vh)).detach().cpu())
            print(f"{n}: {size}, remaining top {top_r} ranks; avg p1 error: {p1_err}, "
                  f"relative p1 error: {relative_err:.2f}%, p2 error {p2_err}")

            s1 = s.detach().cpu().numpy().tolist()
            sum_s = sum(s1)
            for i in range(len(s1)):
                s90.append(s1[i])
                if sum(s90) / sum_s >= 0.90:
                    break
            matrix_profiles[n] = {"size": size,
                                  # "top90_singular_values": s90,
                                  "max_weight": torch.max(torch.max(p)).detach().cpu().numpy().tolist(),
                                  "min_weight": torch.min(torch.min(p)).detach().cpu().numpy().tolist(),
                                  "mean_weight": torch.mean(torch.mean(p)).detach().cpu().numpy().tolist(),
                                  "max_singular_value": max(s90) if len(s90) > 0 else "N/A",
                                  "min_singular_value_90": min(s90) if len(s90) > 0 else "N/A",
                                  "sum_singular_values_90": sum(s90) if len(s90) > 0 else "N/A",
                                  "num_rank_90": len(s90),
                                  "rank90_ratio": len(s90) / min(size),
                                  "compression_factor": compress_factor,
                                  "top_r": top_r,
                                  "avg_p1_error": p1_err,
                                  "relative_p1_error": relative_err,
                                  "p2_error": p2_err,
                                  }

    # save profiles as json files in the folder "matrix_profiles"
    if not os.path.exists("matrix_profiles"):
        os.makedirs("matrix_profiles")
    with open(f"matrix_profiles/{model_name.split('/')[-1]}_{start}_{compress_factor}.json", "w") as f:
        json.dump(matrix_profiles, f, indent=4)
    print(f"matrix profiles saved in matrix_profiles/{model_name.split('/')[-1]}_{start}_{compress_factor}.json")


def compare_mlp():
    from modeling_llama import LlamaMLP

    hidden_size = 1024
    intermediate_size = 4096
    gate_layer = nn.Linear(hidden_size, intermediate_size, bias=False)
    up_layer = nn.Linear(hidden_size, intermediate_size, bias=False)
    down_layer = nn.Linear(intermediate_size, hidden_size, bias=False)
    act_layer = nn.SiLU()

    input_feature = torch.randn(1, 10, hidden_size)
    gate_feature = gate_layer(input_feature)
    act_feature = act_layer(gate_feature)


if __name__ == "__main__":
    profile_llama()

    # m = torch.randn(4096, 4096).to("cuda:0")
    # u, s, vh = torch.linalg.svd(m, full_matrices=True)
    # m_re = u @ (torch.diag(s) @ vh)
    # print(torch.dist(m, m_re))

    # print(torch.dist(torch.zeros(2, 2), torch.ones(2, 2), p=1))
