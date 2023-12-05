from .modeling import (
    replace_linear_with_svd,
    svd_approximation,
)

from .decompose import (
    calc_rank,
    calc_error,
    svd_decomposition,
    weighted_svd_decomposition
)

from .weightcollecter import (
    collect_fisher_info,
    save_ipt_dict,
    load_ipt_dict,
    run_collector,
)

from .mytrainer import (
    clamp_to_fp16_range,
    NoOptimizerTrainer,
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
