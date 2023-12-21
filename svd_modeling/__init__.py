from .modeling import (
    linear_to_svdlinear,
    svd_approximation,
)

from .decompose import (
    calc_rank,
    calc_error,
    svd_decomposition,
    weighted_svd_decomposition
)

from .weightcollector import (
    collect_fisher_info,
    save_fisher_info,
    load_fisher_info,
    run_collector,
)

from .mytrainer import (
    clamp_to_fp16_range,
    NoOptimizerTrainer,
    LoRATrainer,
    LoRASVDLinear,
)

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "down_proj",
    "up_proj",
]
