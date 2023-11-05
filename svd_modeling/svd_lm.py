"""
This file contains the code for the SVD-LM model.
"""
from typing import Dict, Tuple

import numpy as np
import torch.nn
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.activations import SiLUActivation

from utils.visualizer import ProfileVisualizer


class AutoSVDHandler:
    """
    This class is used to load LM from huggingface and conduct SVD on the params.
    Support profiling the weight matrices and activation values by inserting hooks.
    It is tested on Llama, if applied to other models, overwrite functions to fit them.
    """
    def __init__(self, model_name_or_path, derived_model: str = None, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.derived_model = derived_model
        self.cache_dir = kwargs.pop("cache_dir", "../.cache")

        derived_model_dict = {
            "CausalLM": AutoModelForCausalLM,
            "SequenceClassification": AutoModelForSequenceClassification,
        }

        self.model_config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=self.cache_dir, **kwargs)
        self.auto_model_class = derived_model_dict[derived_model] if derived_model else AutoModel
        self.model = None

        # profile related
        self.act_hooks = []
        self.weight_hooks = []
        self.act_maps = {}
        self.grad_maps = {}

    def init_with_empty_weights(self, **kwargs):
        with init_empty_weights():
            self.model = self.auto_model_class.from_config(self.model_config, **kwargs)
        return self.model

    def load_weights(self, **kwargs) -> Tuple[PreTrainedModel, Dict[str, str]]:
        """
        Load the weights of a model from a pretrained model.
        **kwags include dict for device_map
                        args for AutoModel.from_pretrained
        """
        device_map = kwargs.pop("device_map", "auto")
        self.model = self.auto_model_class.from_pretrained(self.model_name_or_path,
                                                           cache_dir=self.cache_dir,
                                                           device_map=device_map,
                                                           **kwargs)
        return self.model, device_map

    def reset_act_maps(self):
        self.act_maps = {}

    def reset_grad_maps(self):
        self.grad_maps = {}

    def reg_hook_for_act(self):
        """
        Profile the activation map.
        1.Register hooks for activation layers.
        2.Collect the activation map.
        3.Do regularization and profile.
        """
        # register hooks
        for name, module in self.model.named_modules():
            # edit the following line to profile other activation layers
            if isinstance(module, SiLUActivation):
                h = module.register_forward_hook(self._get_activation(name))
                self.act_hooks.append(h)

    def del_hook_for_act(self):
        """
        Delete the hooks for activation layers.
        """
        for h in self.act_hooks:
            h.remove()

    def _get_activation(self, name):
        """
        Get the activation map.
        the input shape is [batch_size, seq_len, hidden_size]
        """

        def hook(module, input, output):
            act_out = torch.mean(output.detach(), dim=0).cpu().numpy()
            if name not in self.act_maps:
                self.act_maps[name] = []
            self.act_maps[name].append(act_out)

        return hook

    def profile_act(self, return_act_val=True, percentile: int = 95):
        """
        Profile the activation map.
        """
        act_profiles = {}
        for name, act_list in tqdm(self.act_maps.items(), desc="Processing activations"):
            # act_list is in shape [forward_steps, seq_len, hidden_size]
            act_profiles[name] = {"act_val": act_list if return_act_val else None,
                                  "act_mean": np.mean(act_list, axis=0),
                                  "act_std": np.std(act_list, axis=0),
                                  "act_max": np.max(act_list, axis=0),
                                  "act_min": np.min(act_list, axis=0),
                                  "act_percentile": np.percentile(act_list, percentile, axis=0)
                                  }

        return act_profiles

    def reg_hook_for_weight(self, target_mats=None):
        """
        Profile the weight matrix.
        1.Register hooks for weight tensors.
        2.Collect the gradient.
        3.Do profile.
        """
        # register hooks
        if target_mats is None:
            for name, param in self.model.named_parameters():
                if "weight" in name and "embed" not in name and "layer_norm" not in name:
                    h = param.register_hook(self._get_grad(name))
                    self.weight_hooks.append(h)
        else:
            for name, param in self.model.named_parameters():
                if name in target_mats:
                    h = param.register_hook(self._get_grad(name))
                    self.weight_hooks.append(h)

    def del_hook_for_weight(self):
        """
        Delete the hooks for weight tensors.
        """
        for h in self.weight_hooks:
            h.remove()

    def _get_grad(self, name):
        """
        get the gradient of the weight matrix.
        """

        def hook(grad):
            grad = grad.detach().cpu().numpy()
            if name not in self.grad_maps:
                self.grad_maps[name] = []
            self.grad_maps[name].append(grad)

        return hook

    def profile_grad(self, return_grad_val=True, percentile: int = 95):
        """
        Profile the gradient.
        """
        grad_profiles = {}
        for name, grad_map in tqdm(self.grad_maps.items(), desc="Processing gradients"):
            # grad_map is in shape of weight matrices
            grad_profiles[name] = {"grad_val": grad_map if return_grad_val else None,
                                   "grad_mean": np.mean(grad_map),
                                   "grad_std": np.std(grad_map),
                                   "grad_max": np.max(grad_map),
                                   "grad_min": np.min(grad_map),
                                   "grad_percentile": np.percentile(grad_map, percentile)
                                   }

        return grad_profiles

    def wrap_as_pipeline(self):
        """
        Wrap the model as a pipeline to run benchmark
        """
        ...


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    svd_handler = AutoSVDHandler(model_name, derived_model="CausalLM", cache_dir="../.cache")

    empty_model = svd_handler.init_with_empty_weights()
    # my_device_map = generate_device_map(model, model_config)
    # print(my_device_map)
    model, _ = svd_handler.load_weights()
    print(model)

    svd_handler.reg_hook_for_act()
    svd_handler.reg_hook_for_weight()
    print(len(svd_handler.act_hooks), len(svd_handler.weight_hooks))

    input_ids = torch.randint(0, 32000, (2, 8, 50)).to("cuda:0")
    model.config.pad_token_id = svd_handler.model_config.eos_token_id
    for batch in input_ids:
        output = model.forward(batch)
        # loss = output.logits.mean()
        # loss.backward()

    # act_profiles = svd_handler.profile_act(return_act_val=False)
    # grad_profiles = svd_handler.profile_grad(return_grad_val=False)

    visualizer = ProfileVisualizer()
    # visualizer.draw_heatmap(act_profiles, save=True)
    # visualizer.violinplot(act_profiles, save=True)

    # grad_profiles = {"test": {"grad_val": torch.randn(4096, 4096).numpy()}}
    # visualizer.grad_plot(grad_profiles)
