"""
Run SVD and FWSVD as baseline
"""
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

from evaluate import evaluate_ppl
from svd_modeling import replace_linear_with_svd, TARGET_MODULES, save_ipt_dict, run_collector, load_ipt_dict


def test_svd(
    model_name,
    cache_dir: str = ".cache",
    compress_rate: float = 0.1,
    fp16: bool = False,
):
    # for debug purpose
    config = AutoConfig.from_pretrained(model_name)
    config.update({"num_hidden_layers": 2})

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if fp16:
        model = model.half()
    model.to("cuda")
    replace_linear_with_svd(model.model, compress_rate, TARGET_MODULES, print_info=True)

    if fp16:
        model = model.half()

    ppl = evaluate_ppl(model, tokenizer, "wikitext", "../data")
    print(f"ppl on wikitext-2: {ppl}")


def test_fwsvd(
    model_name,
    cache_dir: str = ".cache",
    weight_dir: str = None,
    compress_rate: float = 0.1,
    fp16: bool = False,
):
    # for debug purpose
    config = AutoConfig.from_pretrained(model_name)
    config.update({"num_hidden_layers": 2})
    # LlamaForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model.to("cuda")
    if weight_dir is None:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir="../data", split='train')
        dataset = dataset.select(range(1000))
        ipt_dic = run_collector(model, tokenizer, dataset, "fisher", 2048, 4, True, False)

        save_dir = cache_dir + "/fisher.pt"
        save_ipt_dict(ipt_dic, save_dir)

    if fp16:
        model = model.half()
    ipt_dic = load_ipt_dict(weight_dir, "cuda")
    for k, v in ipt_dic.items():
        print(k, v.shape, v.max().item(), v.min().item(), "\n")

    replace_linear_with_svd(model.model, compress_rate, TARGET_MODULES, importance_dict=ipt_dic, print_info=True)
    if fp16:
        model = model.half()

    ppl = evaluate_ppl(model, tokenizer, "wikitext", "../data")
    print(f"ppl on wikitext-2: {ppl}")


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    test_fwsvd(model_name, "../.cache", "../.cache/fisher.pt", 0.1, True)
