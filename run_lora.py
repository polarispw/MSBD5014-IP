from experiment import lora_tune


def run_lora(
    model_name,
    dataset_name,
    cache_dir: str = ".cache",
    data_dir: str = ".data",
    save_dir: str = "lora_weights",
):
    lora_tune(
        model_name,
        dataset_name,
        cache_dir=cache_dir,
        data_dir=data_dir,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    import argparse
    import os

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--data_dir", type=str, default=".data")
    parser.add_argument("--save_dir", type=str, default="lora_weights")
    args = parser.parse_args()

    run_lora(
        args.model_name,
        args.dataset_name,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )
