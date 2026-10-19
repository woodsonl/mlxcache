"""Engine adapter for the local `qwen4_exp` (Qwen3.8-Flash-Next) checkpoint.

Selected with `MLXCACHE_ENGINE=qwen4-exp`. Everything about the daemon
protocol — the wire checkpoint, the cache convention, the trust boundary, the
quantization guard, the Metal cache limit, the identity fields — is inherited
from `MlxLmEngine` unchanged. Only CONSTRUCTION differs: the model is not
loaded by mlx-lm's `load()`, because the checkpoint is 132.6 GB and no mlx-lm
architecture knows this model. It is built here from this repository's own
architecture modules over a streaming weight source, then adopted through the
parent's own `_adopt_model` so the post-load invariants have one
implementation.

The adapter is a construction subclass on purpose. Reimplementing the adapter
surface would mean a second copy of the cache convention and the checkpoint
trust boundary, and drift between those two copies is how wrong KV gets served
silently. The substitution seam is `_build` and nothing else.

Cache composition is a cache-identity input (connector protocol §4): a dense
mlx-lm model and this hybrid model must never exchange blobs. `cache_layout`
comes from the inherited probe over THIS model's prompt cache, so the signature
follows the architecture rather than a hand-written constant, and `engine_id`
namespaces the key domain independently of it.
"""

from __future__ import annotations

import json
import os

from .server import MlxLmEngine


class Qwen4ExpEngine(MlxLmEngine):
    """`MlxLmEngine` over this repository's qwen4_exp architecture."""

    # A distinct key domain: a blob written by the generic mlx-lm engine is not
    # interchangeable with this architecture's cache composition, and the fold
    # keeps the two apart even if every other fingerprint field agrees.
    engine_id = "qwen4-exp"

    def __init__(self, model_id: str) -> None:
        # Deliberately not calling super().__init__: that method's first act is
        # mlx-lm's `load`, which cannot build this architecture. The rest of it
        # is reused as-is through `_adopt_model`, so identity and boot guards
        # are the parent's, not a copy.
        self.model_id = model_id
        self.model, self.tokenizer = self._build(model_id)
        self._adopt_model()

    def _build(self, model_id: str):
        """Construct (model, tokenizer) for a local checkpoint directory.

        Imports are lazy for the same reason the parent's `mlx_lm.load` is: the
        hermetic default engine must not pay for this dependency chain.
        """
        from mlxcache_qwen4.loader import CheckpointLoader  # noqa: PLC0415
        from mlxcache_qwen4.model import Qwen4TextConfig, Qwen4TextModel  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        with open(os.path.join(model_id, "config.json"), encoding="utf-8") as fh:
            config_dict = json.load(fh)

        # The loader lives as long as the model: its residency caches (decoded
        # experts, decoded PLE rows) hold the weights the forward pass reads.
        self.loader = CheckpointLoader(model_id)
        model = Qwen4TextModel(Qwen4TextConfig.from_checkpoint(config_dict), self.loader)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
