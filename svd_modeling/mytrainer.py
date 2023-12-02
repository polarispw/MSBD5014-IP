"""
specialized trainer for running importance collection
"""
from typing import Dict, Union, Any, List

import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_utils import ShardedDDPOption
from transformers.utils import is_sagemaker_mp_enabled


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
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [torch.zeros(1)],
                    "weight_decay": self.args.weight_decay,
                }]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                raise NotImplementedError("Sharded DDP Simple is not yet supported for NoOptimizerTrainer.")
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    raise NotImplementedError("Adam8bit is not yet supported for NoOptimizerTrainer.")

        if is_sagemaker_mp_enabled():
            print("Wrapping optimizer inside SMP, not supported.")

        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

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
                p = param.grad.detach().half() if self.save_half else param.grad.detach()
                if self.offload_cpu:
                    p = p.cpu()

                name = name.split(".")
                name = ".".join(name[1:-1])

                if name not in self.ipt_dict:
                    self.ipt_dict[name] = p
                else:
                    self.ipt_dict[name] += p
        self.counter += 1
        return loss

    def get_ipt_dict(self):
        for k, v in self.ipt_dict.items():
            self.ipt_dict[k] = v / self.counter
        return self.ipt_dict
