"""
refer the repo: https://github.com/HanGuo97/lq-lora
"""

import torch
from typing import Tuple, Optional


def calc_rank(
    A: torch.Tensor,
    compress_ratio: float,
) -> int:
    if A.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim}.")

    if compress_ratio < 0 or compress_ratio > 1:
        raise ValueError(f"compress_ratio {compress_ratio} should be in [0, 1].")

    if compress_ratio == 1.0:
        return min(A.shape)

    # compress_ratio = (m + n) * r / (m * n)
    m = A.shape[0]
    n = A.shape[1]
    rank = int(m * n * compress_ratio / (m + n))

    return rank


def calc_error(
    A: torch.Tensor,
    L1: torch.Tensor,
    L2: torch.Tensor,
) -> float:
    if A.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim}.")

    if L1.ndim != 2 or L2.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {L1.ndim} and {L2.ndim}.")

    if L1.shape[1] != L2.shape[0]:
        raise ValueError(f"Expected L1.shape[1] == L2.shape[0], but got {L1.shape[1]} and {L2.shape[0]}.")

    if L1.shape[0] != A.shape[0] or L2.shape[1] != A.shape[1]:
        raise ValueError(f"Expected L1.shape[0] == A.shape[0] and L2.shape[1] == A.shape[1], "
                         f"but got {L1.shape[0]} and {L2.shape[1]}.")

    error = torch.linalg.norm(A - L1 @ L2, ord="fro")
    return error


def svd_decomposition(
    A: torch.Tensor,
    randomized: bool,
    num_ranks: int,
    num_oversampling: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim}.")
    if A.dtype != torch.float32:
        A = A.float()

    if randomized is False:
        U, S, VT = torch.linalg.svd(A, full_matrices=False)
    elif randomized is True:
        num_ranks = min(num_ranks + num_oversampling, min(A.shape))
        U, S, V = torch.svd_lowrank(A, num_ranks)
        # https://pytorch.org/docs/stable/_modules/torch/_lowrank.html#svd_lowrank
        VT = V.mH
    else:
        raise ValueError(f"`randomized` {randomized} not supported")

    S_sqrt = torch.sqrt(S)
    L1 = U * S_sqrt.unsqueeze(dim=0)
    L2 = VT * S_sqrt.unsqueeze(dim=1)
    L1k = L1[:, :num_ranks]
    L2k = L2[:num_ranks, :]
    return L1k, L2k


def weighted_svd_decomposition(
    A: torch.Tensor,
    W: torch.Tensor,
    heuristic: Optional[str],
    num_ranks: int,
    randomized: bool = True,
    num_oversampling: int = 10,
    normalize: bool = False,
    reduce_before_sqrt: bool = True,  # seems to be better empirically
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.ndim != 2 or W.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim} and {W.ndim}.")
    if A.shape != W.shape:
        raise ValueError(f"Expected A.shape == W.shape, but got {A.shape} and {W.shape}.")
    if A.dtype != W.dtype:
        A = A.float()
        W = W.float()
    if A.device != W.device:
        W = W.to(A.device)

    W = torch.clamp(W, min=5.97e-08)  # avoid zero division in line 140+

    if heuristic is None:
        heuristic = "two-sided"

    if normalize is True:
        W = W / torch.linalg.norm(W, ord="fro")

    if reduce_before_sqrt is True:
        # (A.shape[0], 1)
        W1 = torch.sqrt(torch.mean(W, dim=1, keepdim=True))
        # (1, A.shape[1])
        W2 = torch.sqrt(torch.mean(W, dim=0, keepdim=True))
    else:
        # (A.shape[0], 1)
        W1 = torch.mean(torch.sqrt(W), dim=1, keepdim=True)
        # (1, A.shape[1])
        W2 = torch.mean(torch.sqrt(W), dim=0, keepdim=True)

    if heuristic == "none":
        A_tilde = A
    elif heuristic == "row":
        A_tilde = W1 * A
    elif heuristic == "col":
        A_tilde = A * W2
    elif heuristic == "two-sided":
        A_tilde = W1 * A * W2
    else:
        raise NotImplementedError

    L1_tilde, L2_tilde = svd_decomposition(
        A_tilde,
        randomized=randomized,
        num_ranks=num_ranks,
        num_oversampling=num_oversampling
    )

    if heuristic == "none":
        L1 = L1_tilde
        L2 = L2_tilde
    elif heuristic == "row":
        L1 = L1_tilde / W1
        L2 = L2_tilde
    elif heuristic == "col":
        L1 = L1_tilde
        L2 = L2_tilde / W2
    elif heuristic == "two-sided":
        L1 = L1_tilde / W1
        L2 = L2_tilde / W2
    else:
        raise NotImplementedError

    return L1, L2


if __name__ == "__main__":
    from transformers import AutoConfig, AutoModel
    from tqdm import tqdm
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compress-ratio", type=float, default=0.1)
    parser.add_argument("--num-oversampling", type=int, default=5)
    parser.add_argument("--heuristic", type=str, default="two-sided")
    parser.add_argument("--num-runs", type=int, default=10)
    args = parser.parse_args()

    model_name = "meta-llama/Llama-2-7b-hf"
    config = AutoConfig.from_pretrained(model_name)
    config.update({"num_hidden_layers": 2})
    model = AutoModel.from_pretrained(model_name, config=config, cache_dir="../.cache").to("cuda:0")
    print(model)

    A = model.layers[0].mlp.up_proj.weight.detach().clone()
    W = torch.randint_like(A, 0, 2).float().to("cuda:0")
    n_ranks = calc_rank(A, args.compress_ratio)

    print(f"compress_ratio: {args.compress_ratio}, num_ranks: {n_ranks}, "
          f"num_oversampling: {args.num_oversampling}, heuristic: {args.heuristic}, num_runs: {args.num_runs}")

    start = time.time()
    for _ in tqdm(range(args.num_runs)):
        L1, L2 = weighted_svd_decomposition(
            A,
            W,
            heuristic=args.heuristic,
            num_ranks=n_ranks,
        )
    end = time.time()
    print(f"error: {calc_error(A, L1, L2)}")
    print(f"weighted_svd_decomposition: {(end - start) / args.num_runs * 1000:.2f} ms")

    start = time.time()
    for _ in tqdm(range(args.num_runs)):
        L1, L2 = svd_decomposition(
            A,
            randomized=True,
            num_ranks=n_ranks,
            num_oversampling=args.num_oversampling)
    end = time.time()
    print(f"error: {calc_error(A, L1, L2)}")
    print(f"svd_decomposition: {(end - start) / args.num_runs * 1000:.2f} ms")
