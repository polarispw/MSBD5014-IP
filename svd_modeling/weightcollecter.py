"""
Collect the importance matrix of params
"""
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import TrainingArguments

import svd_modeling
from evaluate import process_data, hf_collator
from svd_modeling.mytrainer import NoOptimizerTrainer


def collect_fisher_info(
        model,
        target_params: List[str],
        dataset,
        batch_size: int = 8,
        off_load: bool = False,
        fp16: bool = True,
):
    """
    collect the fisher information of the weight matrix
    """
    args = TrainingArguments(
        output_dir=".",
        overwrite_output_dir=True,
        num_train_epochs=1,

        per_device_train_batch_size=batch_size,
        logging_steps=1,
        save_steps=100000,
        fp16=fp16,
    )

    trainer = NoOptimizerTrainer(
        target_params=target_params,
        save_half=fp16,
        offload_cpu=off_load,
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=hf_collator,
    )

    trainer.train()
    importance_dict = trainer.get_ipt_dict()
    for k, v in importance_dict.items():
        print(k, v.shape, v.max().item(), v.min().item(), "\n")

    return importance_dict


def save_ipt_dict(tensor_dict: Dict[str, torch.Tensor], path: str):
    """
    save the tensor dict
    """
    torch.save(tensor_dict, path)
    print(f"Save the tensor dict to {path}")


def load_ipt_dict(path: str, map_location: str, dtype: torch.dtype = torch.float16):
    """
    load the tensor dict
    """
    dic = torch.load(path, map_location=map_location)
    for k, v in dic.items():
        dic[k] = v.to(dtype)

    return dic


def run_collector(
        model,
        tokenizer,
        dataset,
        collector_type: str,
        seq_len: int = 2048,
        batch_size: int = 8,
        fp16: bool = False,
        off_load: bool = False,
):
    """
    run the importance collector
    """
    # align with huggingface trainer
    dataset = process_data(dataset, tokenizer, seq_len, 'text')

    target_params = svd_modeling.TARGET_MODULES
    if collector_type == "fisher":
        ipt_dic = collect_fisher_info(model, target_params, dataset, batch_size, fp16, off_load)
    else:
        raise NotImplementedError

    return ipt_dic
