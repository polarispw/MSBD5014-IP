import numpy as np
import torch

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from svd_modeling.svd_lm import AutoSVDHandler
from utils.visualizer import ProfileVisualizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_mmlu():
    # load MMLU dataset
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="data", split="train")
    print(train_dataset)

    train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 100)
    selected_dataset = train_dataset.shuffle(seed=42).select(range(800))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=".cache")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = selected_dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=50),
        batched=True)

    print(tokenized_dataset)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataloader = torch.utils.data.DataLoader(tokenized_dataset,
                                             batch_size=8,
                                             shuffle=True,
                                             num_workers=0)

    # load model
    model_name = "meta-llama/Llama-2-7b-hf"
    svd_handler = AutoSVDHandler(model_name, derived_model="CausalLM", cache_dir=".cache")

    empty_model = svd_handler.init_with_empty_weights()
    # my_device_map = generate_device_map(model, model_config)
    # print(my_device_map)
    model, _ = svd_handler.load_weights()
    print(model)

    # setup profiler
    svd_handler.reg_hook_for_act()
    svd_handler.reg_hook_for_weight()
    print(len(svd_handler.act_hooks), len(svd_handler.weight_hooks))

    # run model
    model.to(device)
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model.forward(**batch)
        # loss = output.logits.mean()
        # loss.backward()

    # profile
    act_profiles = svd_handler.profile_act(return_act_val=False)
    grad_profiles = svd_handler.profile_grad(return_grad_val=False)

    visualizer = ProfileVisualizer()
    visualizer.draw_heatmap(act_profiles, save=True)
    visualizer.violinplot(act_profiles, save=True)


if __name__ == "__main__":
    run_mmlu()
