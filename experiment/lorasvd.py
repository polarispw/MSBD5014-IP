import os

import torch
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from evaluate import preprocess_for_causallm, eval_ppl
from svd_modeling import TARGET_MODULES


LANGUAGES_LIST = [
    "ar",
    "zh",
    "en",
    "fr",
    "ja",
    "ko",
    "es",
    "ru",
]


def download_OSCAR(lang_list, sample_num, cache_dir):
    ds_dict = {}
    for language in tqdm(lang_list):
        ds = {"text": []}
        # streamly download samples from OSCAR
        dataset = load_dataset("oscar-corpus/OSCAR-2301", language, cache_dir=cache_dir, split="train", streaming=True)
        for i, sample in enumerate(dataset):
            if i > sample_num:
                break
            ds["text"].append(sample["text"])
        ds = Dataset.from_dict(ds)
        ds_dict[language] = ds

    # save the dict as json
    import json
    with open(os.path.join(cache_dir, "oscar.json"), "w") as f:
        json.dump(ds_dict, f, indent=4)


def lora_tune(
        model_name,
        dataset_name,
        cache_dir: str = ".cache",
        data_dir: str = ".data",
        save_dir: str = "lora_weights",
        quantize: bool = False,
):
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # lora config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_data_dir = os.path.join(data_dir, 'oscar.json')
    train_ds = load_dataset('json', data_files=train_data_dir, field=dataset_name)['train']
    train_ds = train_ds.select(range(2000))
    train_ds = preprocess_for_causallm(train_ds, tokenizer, 2048, 'text', True)

    save_dir = os.path.join(save_dir, dataset_name)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=save_dir,
            overwrite_output_dir=True,
            learning_rate=1e-4,
            num_train_epochs=1,
            max_steps=-1,
            save_strategy="epoch",
            save_only_model=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            logging_steps=4,
            fp16=False,
            gradient_checkpointing=False,
        ),
        train_dataset=train_ds,
    )

    trainer.train()

    eval_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir, split='test')
    eval_ds = preprocess_for_causallm(eval_ds, tokenizer, 2048, 'text')
    ppl = eval_ppl(model, eval_ds)
    print(f"ppl on wikitext-2: {ppl}")


if __name__ == "__main__":
    # collate 8 languages OSCAR dataset
    # download_OSCAR(LANGUAGES_LIST, 5000, ".data")

    # tune Lora on OSCAR dataset ["ar", "zh", "en", "fr", "ja", "ko", "es", "ru"]
    lora_tune(
        "princeton-nlp/Sheared-LLaMA-1.3B",
        "en",
        cache_dir="../.cache",
        data_dir="../.data",
        save_dir="lora_weights",
    )
