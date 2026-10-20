"""Parsed configuration for the `qwen4_exp_text` language model.

The checkpoint's `config.json` is a wrapper whose `text_config` is the text
stack; this dataclass flattens that subtree and validates the invariants the
reference enforces at load time. Construction never touches weights: the
derived tables (n-gram head vocabularies, hash multipliers) are arithmetic on
config fields; the tests cross-check their NAMES and SHAPES against the real
checkpoint's headers (values are not compared).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields
from typing import Any

LINEAR = "linear_attention"
FULL = "full_attention"
# One default for the PLE shard count, shared by the dataclass default and the
# from_checkpoint fallback: the two previously disagreed (128 vs 512), so a
# config.json without the field silently diverged from a direct construction.
DEFAULT_SPLIT_NGRAM_PARTS = 128

# The reference names indexed full-attention layers "qwen_sparse_attention";
# both spellings appear in configs and both mean "full attention + QSA
# indexer" here.
_LAYER_TYPE_ALIASES = {"full_attention": FULL, "qwen_sparse_attention": FULL}


def _as_int(value: Any, default: int | None) -> int | None:
    return default if value is None else int(value)


@dataclass
class Qwen4TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4

    # mRoPE: `rope_parameters` mirrors the checkpoint nesting; the flat fields
    # below are the resolved values the forward pass reads.
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    attention_scaling: float = 1.0

    # Gated delta rule (linear attention).
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    output_gate_type: str | None = None

    # Mixture of experts.
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    num_experts: int = 512
    num_experts_per_tok: int = 10
    norm_topk_prob: bool = True

    # Hyper-connections.
    hc_count: int = 4
    hc_lowrank: int = 320

    # Per-layer embedding / hashed n-grams.
    ple_layer_ids: list[int] = field(default_factory=list)
    ple_embed_dim: int | None = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = DEFAULT_SPLIT_NGRAM_PARTS

    # QSA indexer on full-attention layers.
    indexer_n_heads: int | None = 4
    indexer_kv_heads: int | None = 1
    indexer_head_dim: int | None = 128
    indexer_budget: int | None = 2048
    indexer_compress_ratio: int | None = 4

    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None

    def __post_init__(self) -> None:
        self.ple_layer_ids = sorted({int(i) for i in (self.ple_layer_ids or [])})
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        self.mrope_section = [int(x) for x in (self.mrope_section or [])]
        self._resolve_layer_types()
        self._validate()

    # ------------------------------------------------------------------ derived

    def _resolve_layer_types(self) -> None:
        """Normalise layer types and reject a contradictory interval.

        The interval is the pattern generator: layer `i` is full attention iff
        `(i + 1) % full_attention_interval == 0`. An explicit `layer_types`
        list that disagrees with that pattern describes a different
        architecture than the one the interval claims, so it is an error
        rather than a preference: silently honouring either side would make
        the cache layout and the attention math disagree.
        """
        interval = int(self.full_attention_interval)
        if interval <= 0:
            raise ValueError(f"full_attention_interval must be positive, got {interval}.")
        expected = [LINEAR if (i + 1) % interval else FULL for i in range(self.num_hidden_layers)]
        provided = list(self.layer_types or [])
        if not provided:
            self.layer_types = expected
            return
        if len(provided) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(provided)} entries but num_hidden_layers is "
                f"{self.num_hidden_layers}."
            )
        unknown = sorted(set(provided) - set(_LAYER_TYPE_ALIASES) - {LINEAR})
        if unknown:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {unknown}.")
        normalised = [_LAYER_TYPE_ALIASES.get(t, t) for t in provided]
        mismatch = [
            i for i, (got, want) in enumerate(zip(normalised, expected, strict=True)) if got != want
        ]
        if mismatch:
            raise ValueError(
                "layer_types contradicts full_attention_interval="
                f"{interval}: layer(s) {mismatch[:8]} are "
                f"{[normalised[i] for i in mismatch[:8]]}, the interval pattern requires "
                f"{[expected[i] for i in mismatch[:8]]}."
            )
        self.layer_types = normalised

    def _validate(self) -> None:
        gate = self.output_gate_type or self.hidden_act
        if gate not in {"sigmoid", "silu"}:
            raise ValueError(f"Unsupported Qwen4-Exp output gate activation: {gate}.")
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {self.hc_count}.")
        if self.hc_lowrank <= 0:
            raise ValueError(f"hc_lowrank must be positive, got {self.hc_lowrank}.")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}.")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError(
                "num_experts_per_tok must be in [1, num_experts], got "
                f"{self.num_experts_per_tok} and {self.num_experts}."
            )
        if self.moe_intermediate_size <= 0 or self.shared_expert_intermediate_size <= 0:
            raise ValueError(
                "moe_intermediate_size and shared_expert_intermediate_size must be > 0."
            )
        if self.hidden_size % self.hc_count:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must divide the hyper-connection width."
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                f"linear_num_value_heads ({self.linear_num_value_heads}) must be divisible by "
                f"linear_num_key_heads ({self.linear_num_key_heads})."
            )
        self._validate_qsa()
        self._validate_ple()

    def _validate_qsa(self) -> None:
        qsa_fields = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        values = {name: getattr(self, name) for name in qsa_fields}
        if not any(value is not None for value in values.values()):
            has_full = FULL in self.layer_types
            if has_full:
                raise ValueError("full-attention layers require the QSA indexer fields.")
            return
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}.")
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError(f"QSA config values must be positive: {values}.")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1.")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio.")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim > self.indexer_head_dim:
            raise ValueError(
                "Qwen4-Exp attention RoPE dimensions must fit the QSA index head: "
                f"rotary_dim={rotary_dim}, indexer_head_dim={self.indexer_head_dim}."
            )

    def _validate_ple(self) -> None:
        if not self.ple_layer_ids:
            return
        if self.eos_token_id is None or (
            isinstance(self.eos_token_id, list) and not self.eos_token_id
        ):
            raise ValueError("eos_token_id must be set when Qwen4-Exp PLE layers are enabled.")
        heads = self.ngram_heads
        if heads <= 0 or self.ple_embed_dim <= 0 or self.ple_embed_dim % heads:
            raise ValueError(
                "ple_embed_dim must be positive and divisible by the number of n-gram heads: "
                f"{self.ple_embed_dim} % {heads} != 0."
            )
        bad = [i for i in self.ple_layer_ids if i < 1 or i > self.num_hidden_layers]
        if bad:
            raise ValueError(
                f"ple_layer_ids must contain one-indexed ids in [1, {self.num_hidden_layers}], "
                f"got {bad}."
            )
        non_linear = [i for i in self.ple_layer_ids if self.layer_types[i - 1] != LINEAR]
        if non_linear:
            raise ValueError(
                f"Qwen4-Exp PLE is only supported on linear_attention layers, "
                f"got PLE on {non_linear}."
            )
        if self.ple_conv_kernel_size < 1 or self.ngram_size < 2:
            raise ValueError("ple_conv_kernel_size must be >= 1 and ngram_size >= 2 with PLE.")

    # ------------------------------------------------------------- checkpoint

    @classmethod
    def from_checkpoint(cls, config_dict: dict) -> Qwen4TextConfig:
        # Omitted checkpoint fields take the DATACLASS defaults, read from the
        # dataclass itself — duplicated literals here previously disagreed
        # with the defaults (128 vs 512 in split_ngram_parts, 32768 vs
        # 262144 in max_position_embeddings, 10000 vs 10_000_000 in
        # rope_theta), so a config.json missing a field silently built a
        # different model than a direct construction.
        d = {f.name: f.default for f in fields(cls) if f.default is not dataclasses.MISSING}
        """Build from the parsed `config.json` dict (wrapper or text subtree)."""
        # Single source of defaults: every scalar fallback below reads the
        # dataclass field default via _default(), never a second hand-typed
        # literal. Two disagreeing default sets meant a config.json missing a
        # field built a different architecture depending on which set won
        # (and partial_rotary_factor 1.0 vs 0.25 failed QSA validation).

        def _default(name: str):
            import dataclasses

            for f in dataclasses.fields(cls):
                if f.name == name:
                    if f.default is not dataclasses.MISSING:
                        return f.default
                    if f.default_factory is not dataclasses.MISSING:
                        # Factory fields (mrope_section, layer_types): the
                        # factory IS the default — a literal copy here would
                        # be a second source of truth that can drift.
                        return f.default_factory()
                    raise ValueError(f"{name} has no default")
            raise ValueError(f"unknown field {name}")

        raw = config_dict.get("text_config", config_dict)
        if raw.get("model_type", "qwen4_exp_text") != "qwen4_exp_text":
            raise ValueError(f"unsupported model_type: {raw.get('model_type')!r}")
        rope = raw.get("rope_parameters") or {}
        partial = rope.get(
            "partial_rotary_factor",
            raw.get("partial_rotary_factor", _default("partial_rotary_factor")),
        )
        layer_types = raw.get("layer_types") or _default("layer_types")
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            # DERIVED fallback, not a second default: when the field is absent
            # the checkpoint layout itself fixes head_dim at hidden_size //
            # heads (the dataclass default 256 describes this checkpoint's
            # value, not a general constant), so deriving beats importing it.
            head_dim=int(raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]),
            hidden_act=raw.get("hidden_act", _default("hidden_act")),
            rms_norm_eps=float(raw.get("rms_norm_eps", _default("rms_norm_eps"))),
            max_position_embeddings=int(
                raw.get("max_position_embeddings", _default("max_position_embeddings"))
            ),
            attention_bias=bool(raw.get("attention_bias", _default("attention_bias"))),
            layer_types=list(layer_types),
            full_attention_interval=int(
                raw.get("full_attention_interval", _default("full_attention_interval"))
            ),
            rope_theta=float(rope.get("rope_theta", raw.get("rope_theta", _default("rope_theta")))),
            partial_rotary_factor=float(partial),
            mrope_section=list(rope.get("mrope_section", _default("mrope_section"))),
            linear_conv_kernel_dim=int(
                raw.get("linear_conv_kernel_dim", _default("linear_conv_kernel_dim"))
            ),
            linear_key_head_dim=int(
                raw.get("linear_key_head_dim", _default("linear_key_head_dim"))
            ),
            linear_value_head_dim=int(
                raw.get("linear_value_head_dim", _default("linear_value_head_dim"))
            ),
            linear_num_key_heads=int(
                raw.get("linear_num_key_heads", _default("linear_num_key_heads"))
            ),
            linear_num_value_heads=int(
                raw.get("linear_num_value_heads", _default("linear_num_value_heads"))
            ),
            output_gate_type=raw.get("output_gate_type"),
            moe_intermediate_size=int(
                raw.get("moe_intermediate_size", _default("moe_intermediate_size"))
            ),
            shared_expert_intermediate_size=int(
                raw.get(
                    "shared_expert_intermediate_size", _default("shared_expert_intermediate_size")
                )
            ),
            num_experts=int(raw.get("num_experts", _default("num_experts"))),
            num_experts_per_tok=int(
                raw.get("num_experts_per_tok", _default("num_experts_per_tok"))
            ),
            norm_topk_prob=bool(raw.get("norm_topk_prob", _default("norm_topk_prob"))),
            hc_count=int(raw.get("hc_count", _default("hc_count"))),
            hc_lowrank=int(raw.get("hc_lowrank", _default("hc_lowrank"))),
            ple_layer_ids=list(raw.get("ple_layer_ids") or []),
            ple_embed_dim=_as_int(raw.get("ple_embed_dim"), None),
            ple_conv_kernel_size=int(
                raw.get("ple_conv_kernel_size", _default("ple_conv_kernel_size"))
            ),
            ngram_size=int(raw.get("ngram_size", _default("ngram_size"))),
            heads_per_ngram=int(raw.get("heads_per_ngram", _default("heads_per_ngram"))),
            ngram_vocab_size_base=int(
                raw.get("ngram_vocab_size_base", _default("ngram_vocab_size_base"))
            ),
            make_ngram_vocab_size_divisible_by=int(
                raw.get(
                    "make_ngram_vocab_size_divisible_by",
                    _default("make_ngram_vocab_size_divisible_by"),
                )
            ),
            seed=int(raw.get("seed", _default("seed"))),
            split_ngram_parts=int(raw.get("split_ngram_parts", DEFAULT_SPLIT_NGRAM_PARTS)),
            indexer_n_heads=_as_int(raw.get("indexer_n_heads"), _default("indexer_n_heads")),
            indexer_kv_heads=_as_int(raw.get("indexer_kv_heads"), _default("indexer_kv_heads")),
            indexer_head_dim=_as_int(raw.get("indexer_head_dim"), _default("indexer_head_dim")),
            indexer_budget=_as_int(raw.get("indexer_budget"), _default("indexer_budget")),
            indexer_compress_ratio=_as_int(
                raw.get("indexer_compress_ratio"), _default("indexer_compress_ratio")
            ),
            eos_token_id=raw.get("eos_token_id"),
            bos_token_id=raw.get("bos_token_id"),
        )

    # ---------------------------------------------------------------- geometry

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def hc_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_head_dim(self) -> int:
        """Width of one hashed n-gram head inside the concatenated embedding."""
        return self.ple_embed_dim // self.ngram_heads

    def head_vocab_sizes(self, rank: int) -> list[int]:
        """Prime vocabulary size per hashed head of ONE PLE layer.

        Every PLE layer owns a private table, so offsets restart at zero, but
        the prime vocabularies advance globally by `rank * ngram_heads` so two
        PLE layers never hash into each other's rows.
        """
        base = rank * self.ngram_heads
        return [
            _nth_prime_after(self.ngram_vocab_size_base - 1, base + head + 1)
            for head in range(self.ngram_heads)
        ]

    def head_offsets(self, rank: int) -> list[int]:
        offsets: list[int] = []
        total = 0
        for size in self.head_vocab_sizes(rank):
            offsets.append(total)
            total += size
        return offsets

    def ngram_vocab_total(self, rank: int) -> int:
        """Rows of one PLE layer's n-gram table, padded to the divisor."""
        total = sum(self.head_vocab_sizes(rank))
        divisor = self.make_ngram_vocab_size_divisible_by
        return math.ceil(total / divisor) * divisor

    def ngram_rows_per_shard(self, rank: int) -> int:
        """Rows in each stored shard of one PLE layer's n-gram table.

        The table is stored in `split_ngram_parts` equally-sized shards, so the
        last shard's tail is padding rows the hash never addresses.
        """
        return math.ceil(self.ngram_vocab_total(rank) / self.split_ngram_parts)

    def ple_rank(self, layer_idx: int) -> int | None:
        """Zero-based PLE layer rank for a decoder layer, or None if it has none."""
        one_indexed = layer_idx + 1
        if one_indexed not in self.ple_layer_ids:
            return None
        return self.ple_layer_ids.index(one_indexed)

    @property
    def eos_token(self) -> int:
        return self.eos_token_id[0] if isinstance(self.eos_token_id, list) else self.eos_token_id


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> list[int]:
    """Deterministic per-n-gram-order hash multipliers (odd, 63-bit safe)."""
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + 10007 * ple_layer_index
    return [
        2 * (_splitmix64((base_seed + 0x9E3779B97F4A7C15 * (i + 1)) & ((1 << 64) - 1)) % half_bound)
        + 1
        for i in range(ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime
