"""MLX implementation of the `qwen4_exp` text stack (Qwen3.8-Flash-Next).

The public surface is three names:

    Qwen4TextConfig.from_checkpoint(config_dict)   # parse config.json
    Qwen4TextModel(config, weights)                # build from a WeightSource
    WeightSource                                   # the loader protocol

`Qwen4MTPModel` and `mtp_weight_shapes` cover the multi-token-prediction head,
and `weight_shapes` enumerates every checkpoint name the text stack requests.
"""

from .config import FULL, LINEAR, Qwen4TextConfig
from .model import (
    EMBED_NAME,
    LAYER_PREFIX,
    LM_HEAD_NAME,
    Qwen4MTPModel,
    Qwen4TextModel,
    WeightSource,
    mtp_weight_shapes,
    weight_shapes,
)

__all__ = [
    "EMBED_NAME",
    "FULL",
    "LAYER_PREFIX",
    "LINEAR",
    "LM_HEAD_NAME",
    "Qwen4MTPModel",
    "Qwen4TextConfig",
    "Qwen4TextModel",
    "WeightSource",
    "mtp_weight_shapes",
    "weight_shapes",
]
