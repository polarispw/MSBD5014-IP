"""
Collect the importance matrix of params
"""
from typing import Dict, List, Tuple

import torch
from transformers import TrainingArguments

import svd_modeling
from evaluate import preprocess_for_causallm
from svd_modeling.mytrainer import NoOptimizerTrainer, LoRATrainer


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
    )

    trainer.train()
    importance_dict = trainer.get_ipt_dict()

    return importance_dict


def save_fisher_info(tensor_dict: Dict[str, torch.Tensor], path: str):
    """
    save the tensor dict
    """
    torch.save(tensor_dict, path)
    print(f"Save the tensor dict to {path}")


def load_fisher_info(path: str, dtype: torch.dtype, device: torch.device = "cpu"):
    """
    load the tensor dict
    """
    precision_rank = {"torch.float16": 1, "torch.float32": 2}
    dic = torch.load(path, map_location=device)

    if dtype != dic[list(dic.keys())[0]].dtype:
        for k, v in dic.items():
            # avoid zero division in weighted_svd_decomposition
            if precision_rank[str(dtype)] < precision_rank[str(v.dtype)]:
                v = torch.clamp(v, min=5.97e-08)
            dic[k] = v.to(dtype)

    return dic


def collect_lora_info(
        model,
        dataset,
        batch_size,
        peft_config,
):
    args = TrainingArguments(
        output_dir="./lora_weights",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        logging_steps=4,
        save_strategy="epoch",
        fp16=False,
        gradient_checkpointing=False,
    )

    trainer = LoRATrainer(
        model=model,
        peft_config=peft_config,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()


def load_lora_info(
        model,
        target_params,
        dir_list: List,
):
    # init weight dict
    ipt_dict = {}
    for name, param in model.named_parameters():
        if any([target in name for target in target_params]):
            # layers.###[.weight]
            name = name[:-7]
            ipt_dict[name] = []
            print(name)

    # collect lora weights from dir_list
    prefix = "base_model.model.model."
    for lora_dir in dir_list:
        lora = torch.load(lora_dir)
        # for n in lora.keys():
        #     print(n)
        for name in ipt_dict.keys():
            name1 = prefix + name + ".lora_A.weight"
            name2 = prefix + name + ".lora_B.weight"
            ipt_dict[name].append((lora[name1].T, lora[name2].T))

    return ipt_dict


# will be deprecated after experiment codes finished
def run_collector(
        model,
        tokenizer,
        dataset,
        collector_type: str,
        seq_len: int = 2048,
        batch_size: int = 8,
        half_ipt: bool = False,
        off_load: bool = True,
) -> [Dict[str, torch.Tensor], str]:
    """
    run the importance collector
    """
    # align with huggingface trainer
    dataset = preprocess_for_causallm(dataset, tokenizer, seq_len, 'text')

    target_params = svd_modeling.TARGET_MODULES
    ipt_dic = {}
    if collector_type == "fisher":
        ipt_dic = collect_fisher_info(model, target_params, dataset, batch_size, half_ipt, off_load)
    elif collector_type == "lora":
        collect_lora_info(model, dataset, batch_size)
    else:
        raise NotImplementedError

    return ipt_dic
