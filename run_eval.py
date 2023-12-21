"""
Evaluate the model's performance on the downstream task.
"""
from experiment import (
    test_vanilla_pll,
    test_svd_pll,
    test_fwsvd_pll,
)


def run_pll_baseline(
    test_type: str,
    model_name: str,
    cache_dir: str = ".cache",
    data_dir: str = "data",
    iptdict_dir: str = None,
    compress_rate: float = 0.1,
    fine_tune: bool = False,
    half_model: bool = False,
    half_ipt: bool = False,
):
    """
    run the baseline
    """
    if test_type == "vanilla":
        test_vanilla_pll(
            model_name,
            cache_dir=cache_dir,
            data_dir=data_dir,
            half_model=half_model,
        )
    elif test_type == "svd":
        test_svd_pll(
            model_name,
            cache_dir=cache_dir,
            data_dir=data_dir,
            compress_rate=compress_rate,
            fine_tune=fine_tune,
            half_model=half_model,
        )
    elif test_type == "fwsvd":
        test_fwsvd_pll(
            model_name,
            cache_dir=cache_dir,
            data_dir=data_dir,
            weight_dir=iptdict_dir,
            compress_rate=compress_rate,
            fine_tune=fine_tune,
            half_model=half_model,
            half_ipt=half_ipt,
        )
    else:
        raise NotImplementedError(f"test type {test_type} not implemented")


if __name__ == "__main__":
    import argparse

    # "meta-llama/Llama-2-7b-hf"
    # "princeton-nlp/Sheared-LLaMA-1.3B"

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_type", type=str, default="vanilla")
    parser.add_argument("--model_name", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--data_dir", type=str, default=".data")
    parser.add_argument("--iptdict_dir", type=str, default=".cache/llama1b_fisher.pt")
    parser.add_argument("--compress_rate", type=float, default=0.1)
    parser.add_argument("--fine_tune", type=bool, default=True)
    parser.add_argument("--half_model", type=bool, default=True)
    parser.add_argument("--half_ipt", type=bool, default=True)
    args = parser.parse_args()

    run_pll_baseline(
        args.test_type,
        args.model_name,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        iptdict_dir=args.iptdict_dir,
        compress_rate=args.compress_rate,
        fine_tune=args.fine_tune,
        half_model=args.half_model,
        half_ipt=args.half_ipt,
    )
