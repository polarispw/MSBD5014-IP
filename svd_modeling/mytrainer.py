"""
Specialized trainer for running importance collection
"""
from typing import Dict, Union, Any, List

import torch
from torch import nn
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled

from svd_modeling.modeling import LoRASVDLinear


def clamp_to_fp16_range(tensor):
    """
    Clamp the values of a tensor to the representable range of fp16.
    """
    min_val = torch.tensor(5.97e-08).to(tensor.device)
    max_val = torch.tensor(65500).to(tensor.device)
    return torch.clamp(tensor, min=min_val, max=max_val)


class NoOptimizerTrainer(Trainer):
    def __init__(self, target_params: List[str], save_half: bool, offload_cpu: bool, **kwargs):
        super().__init__(**kwargs)
        self.target_params = target_params
        self.save_half = save_half
        self.offload_cpu = offload_cpu
        self.ipt_dict = {}
        self.counter = 0

    def create_optimizer(self):
        """
        It is hard to reload the _inner_training_loop to disable the optimizer.
        So we use a dummy optimizer with no params inside to replace the original one.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        flag = torch.zeros(1).to(opt_model.device)
        flag.requires_grad = True
        flag.grad = torch.zeros(1).to(flag.device)
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [flag],
                    "weight_decay": self.args.weight_decay,
                }]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                raise NotImplementedError("Adam8bit is not supported yet.")

        if is_sagemaker_mp_enabled():
            raise NotImplementedError("SageMaker Model Parallelism is not supported yet.")

        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Add collection part after the original training step.
        Note that gradients on well-trained model are very small, while some are salient.
        So we use the abs instead of sqr and be careful with the overflow when using fp16.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        loss = Trainer.training_step(self, model, inputs)

        for name, param in model.named_parameters():
            if any([target in name for target in self.target_params]):
                w = param.grad.detach().abs()
                if self.save_half:
                    w = clamp_to_fp16_range(w)
                    w = w.half()
                if self.offload_cpu:
                    w = w.cpu()

                name = name.split(".")
                name = ".".join(name[1:-1])

                if name not in self.ipt_dict:
                    self.ipt_dict[name] = w
                else:
                    self.ipt_dict[name] = (self.ipt_dict[name]*self.counter + w)/(self.counter+1)

        self.counter += 1
        torch.cuda.empty_cache()
        return loss

    def get_ipt_dict(self):
        return self.ipt_dict


class LoRASVDTrainer(Trainer):
    """
    for fine-tuning LoRA weighted SVD model
    """
    def __init__(self, lora_tuning: bool, **kwargs):
        super().__init__(**kwargs)
        if lora_tuning:
            print("##### Tuning LoRA module, keep stem frozen #####")
            for n, m in self.model.named_modules():
                if isinstance(m, LoRASVDLinear):
                    m.freeze_stem()
