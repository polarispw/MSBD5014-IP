"""
Collect the importance matrix of params
"""
from typing import Dict, List

import torch
from transformers import TrainingArguments

import svd_modeling
from evaluate import process_data, hf_collator
from svd_modeling.mytrainer import NoOptimizerTrainer


def collect_fisher_info(
        model,
        target_params: List[str],
        dataset,
        batch_size: int = 4,
        half_ipt: bool = True,
        off_load: bool = True,
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
        save_steps=1e8,
        fp16=False,
        gradient_checkpointing=False,
    )

    trainer = NoOptimizerTrainer(
        target_params=target_params,
        save_half=half_ipt,
        offload_cpu=off_load,
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=hf_collator,
    )

    trainer.train()
    importance_dict = trainer.get_ipt_dict()

    return importance_dict


def save_ipt_dict(tensor_dict: Dict[str, torch.Tensor], path: str):
    """
    save the tensor dict
    """
    torch.save(tensor_dict, path)
    print(f"Save the tensor dict to {path}")


def load_ipt_dict(path: str, dtype: torch.dtype):
    """
    load the tensor dict
    """
    dic = torch.load(path)
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
        half_ipt: bool = False,
        off_load: bool = True,
):
    """
    run the importance collector
    """
    # align with huggingface trainer
    dataset = process_data(dataset, tokenizer, seq_len, 'text')

    target_params = svd_modeling.TARGET_MODULES
    ipt_dic = {}
    if collector_type == "fisher":
        ipt_dic = collect_fisher_info(model, target_params, dataset, batch_size, half_ipt, off_load)
    elif collector_type == "lora":
        ...
    else:
        raise NotImplementedError

    return ipt_dic
