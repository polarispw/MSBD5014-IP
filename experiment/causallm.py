"""
Run SVD and FWSVD as baseline
"""
import os.path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from evaluate import evaluate_ppl, preprocess_for_causallm, eval_ppl
from svd_modeling import (
    TARGET_MODULES,
    linear_to_svdlinear,
    svd_approximation,
    save_fisher_info,
    run_collector,
    load_fisher_info,
)
from svd_modeling.mytrainer import LoRASVDTrainer
from svd_modeling.weightcollector import load_lora_info


def test_vanilla(
    model_name,
    cache_dir: str = ".cache",
    data_dir: str = ".data",
    half_model: bool = False,
):
    # for locally debug
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    # config.update({"num_hidden_layers": 2})

    torch.manual_seed(42)
    model_dtype = torch.float32 if not half_model else torch.float16
    device = f"cuda:{torch.cuda.current_device()}"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
        device_map=device,
        torch_dtype=model_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir, split='test')
    dataset = dataset.select(range(200))
    dataset = preprocess_for_causallm(dataset, tokenizer, 2048, 'text')
    ppl = eval_ppl(model, dataset)
    print(f"ppl on wikitext-2: {ppl}")


def test_svd(
    model_name,
    cache_dir: str = ".cache",
    data_dir: str = ".data",
    compress_rate: float = 0.1,
    fine_tune: bool = False,
    half_model: bool = False,
):
    # for locally debug
    config = AutoConfig.from_pretrained(model_name)
    # config.update({"num_hidden_layers": 2})

    torch.manual_seed(42)
    model_dtype = torch.float32 if not half_model else torch.float16
    device = f"cuda:{torch.cuda.current_device()}"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
        device_map=device,
        torch_dtype=model_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if not fine_tune:
        linear_to_svdlinear(model.model, compress_rate, TARGET_MODULES, print_info=True)
    else:
        # decompose then reconstruct
        svd_approximation(model.model, compress_rate, TARGET_MODULES, print_info=True)

        # LoRA fine-tuning
        ft_dir = os.path.join(cache_dir, "ft_results")
        resume_dir = ".cache/ft_results/checkpoint-19"
        if os.path.exists(resume_dir):
            print("Loading from checkpoint")
            config = PeftConfig.from_pretrained(resume_dir)
            model = get_peft_model(model, config)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=TARGET_MODULES,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir, split='train')
            dataset = dataset.select(range(2000))
            dataset = preprocess_for_causallm(dataset, tokenizer, 2048, 'text')

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=ft_dir,
                    overwrite_output_dir=True,
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    logging_steps=4,
                    save_strategy="epoch",
                    fp16=False,
                    gradient_checkpointing=False,
                ),
                train_dataset=dataset,
            )

            trainer.train()

    print(model)
    ppl = evaluate_ppl(model, tokenizer, "wikitext", data_dir)
    print(f"ppl on wikitext-2: {ppl}")


def test_fwsvd(
    model_name,
    cache_dir: str = ".cache",
    data_dir: str = ".data",
    weight_dir: str = None,
    compress_rate: float = 0.1,
    fine_tune: bool = False,
    half_model: bool = False,
    half_ipt: bool = False,
):
    # for locally debug
    config = AutoConfig.from_pretrained(model_name)
    # config.update({"num_hidden_layers": 2})

    torch.manual_seed(42)
    model_dtype = torch.float32 if not half_model else torch.float16
    device = f"cuda:{torch.cuda.current_device()}"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
        device_map=device,
        torch_dtype=model_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if weight_dir is None:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir, split='train')
        dataset = dataset.select(range(2000))
        ipt_dic = run_collector(
            model,
            tokenizer,
            dataset,
            collector_type="fisher",
            seq_len=2048,
            batch_size=4,
            half_ipt=half_ipt,
            off_load=True
        )

        save_dir = cache_dir + "/fisher.pt"
        save_fisher_info(ipt_dic, save_dir)

        weight_dir = save_dir

    ipt_dtype = torch.float32 if not half_ipt else torch.float16
    ipt_dic = load_fisher_info(weight_dir, ipt_dtype)
    for k, v in ipt_dic.items():
        print(k, v.shape, v.max().item(), v.min().item(), "\n")

    if not fine_tune:
        linear_to_svdlinear(model.model, compress_rate, TARGET_MODULES, ipt_dict=ipt_dic, print_info=True)
    else:
        # decompose then reconstruct
        svd_approximation(model.model, compress_rate, TARGET_MODULES, importance_dict=ipt_dic, print_info=True)

        # LoRA fine-tuning
        ft_dir = os.path.join(cache_dir, "ft_results")
        resume_dir = ".cache/ft_results/checkpoint-19"
        if os.path.exists(resume_dir):
            print("Loading from checkpoint")
            config = PeftConfig.from_pretrained(resume_dir)
            model = get_peft_model(model, config)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=TARGET_MODULES,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir, split='train')
            dataset = dataset.select(range(2000))
            dataset = preprocess_for_causallm(dataset, tokenizer, 2048, 'text')

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=ft_dir,
                    overwrite_output_dir=True,
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    logging_steps=4,
                    save_strategy="epoch",
                    fp16=False,
                    gradient_checkpointing=False,
                ),
                train_dataset=dataset,
            )

            trainer.train()

    ppl = evaluate_ppl(model, tokenizer, "wikitext", data_dir)
    print(f"ppl on wikitext-2: {ppl}")


def test_lwsvd(
    model_name,
    cache_dir: str = ".cache",
    data_dir: str = ".data",
    weight_dir: str = None,
    compress_rate: float = 0.1,
    fine_tune: bool = False,
    half_model: bool = False,
    half_ipt: bool = False,
):
    # for locally debug
    config = AutoConfig.from_pretrained(model_name)
    # config.update({"num_hidden_layers": 2})

    torch.manual_seed(42)
    model_dtype = torch.float32 if not half_model else torch.float16
    device = f"cuda:{torch.cuda.current_device()}"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
        device_map=device,
        torch_dtype=model_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    ipt_dic = load_lora_info(model.model, TARGET_MODULES, [".cache/checkpoint-94/adapter_model.bin"])
    linear_to_svdlinear(model.model, compress_rate, TARGET_MODULES, ipt_dict=ipt_dic, print_info=True)
    print(model)
    if not fine_tune:
        ...
    else:
        train_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir, split='train')
        train_ds = train_ds.select(range(2000))
        train_ds = preprocess_for_causallm(train_ds, tokenizer, 2048, 'text')

        LoRASVDTrainer(
            lora_tuning=True,
            model=model,
            dataset_name=train_ds,
            args=TrainingArguments(
                output_dir=data_dir,
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                logging_steps=4,
                save_strategy="epoch",
                fp16=False,
                gradient_checkpointing=False,
            ),
        )

    test_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir, split='test')
    test_ds = preprocess_for_causallm(test_ds, tokenizer, 2048, 'text')

    ppl = eval_ppl(model, test_ds)
    print(f"ppl on wikitext-2: {ppl}")


if __name__ == "__main__":
    model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
    # data_dir = "../.data"
    # test_vanilla(model_name, "../.cache", data_dir=data_dir, half_model=True)
    # test_svd(model_name, "../.cache", data_dir=data_dir, compress_rate=0.2, half_model=True)
    # test_fwsvd(
    #     model_name,
    #     cache_dir="../.cache",
    #     data_dir=data_dir,
    #     weight_dir="../.cache/llama1b_fisher.pt",
    #     compress_rate=0.1,
    #     half_model=True
    # )
