"""Streaming weight loader for the Qwen3.8-Flash-Next-NVFP4 checkpoint.

The checkpoint is 132.6 GB and the target machine has 48 GB of RAM, so the
loader must never hold more than a caller-requested tensor (or row range).
Two structures dominate: 70.5 GB of routed NVFP4 experts (48 layers x 512
experts) and 51.2 GB of PLE n-gram rows split across 128 FP8 shards — the
same figures, in the same GB units, as the tables in docs/ops.md and README.

Dequantisation recipes, each chosen from measurement rather than assumption
----------------------------------------------------------------------------

NVFP4 (``model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj``;
``weight`` is U8 holding packed 4-bit values, ``weight_scale`` is F8_E4M3 with
``group_size=16``, ``weight`` is ``[out, in/2]`` and ``weight_scale`` is
``[out, in/16]``)::

    wq    = weight_bytes.view(uint32).reshape(out, in // 8)   # eight 4-bit codes per word
    scale = weight_scale bytes, RAW uint8, [out, in // 16]    # mlx decodes e4m3 itself
    gs    = per-expert scalar, float32, from
            model.language_model.layers.N.mlp.switch_mlp.{proj}.global_scale[E]
    out   = mx.dequantize(wq, scales=scale, group_size=16, bits=4,
                          mode="nvfp4", global_scale=gs, dtype=bfloat16)

``global_scale`` must be the per-expert table entry, not ``weight_scale_2``.
Measured on layer 0 / expert 0 / gate_proj against that layer's BF16
``mlp.shared_expert.gate_proj`` (rms 0.00993), RMS of the dequantised tensor:

    global_scale omitted             rms 204.0     ratio 20551   (wrong)
    global_scale = weight_scale_2    rms 5.0e-06   ratio 0.00055 (wrong, 2688x small)
    global_scale = per-expert table  rms 0.01349   ratio 1.359   (chosen)

Across 20 (layer, expert) samples spanning layers 0/1/5/23/47 the ratio of
mean|dequantised| to mean|shared expert| is 1.12..1.61 with mean 1.29, i.e.
the same order of magnitude, which is what a routed expert should look like;
the two rejected candidates are off by 2-5 orders of magnitude. Independent
cross-check: ``6 * 448 * weight_scale_2`` equals the per-expert global_scale
(448 is the e4m3 maximum, 6 the e2m1 maximum), ratio 1.0000 for gate, up and
down on every layer checked; that identity is asserted as a test. A hand
unpack of the nibbles decoded as E2M1 times the expanded e4m3 group scale
times ``weight_scale_2`` reproduces mlx's ``global_scale`` output to a
relative error of 9.5e-4, so the two descriptions are the same weights.

FP8 (any ``*_scale_inv`` pair)::

    out = mx.from_fp8(weight_bytes, float32) * expand(weight_scale_inv)

The block layout was determined by measuring, for every (out, in) block, the
largest FP8 magnitude inside it. With ``weight_scale_inv`` read as shaped
(``[out/128, in/128]`` here) every block of every projection has amax exactly
448.0 -- the E4M3 maximum, i.e. each block is normalised to fill its range.
The transposed reading yields uneven block amax (240.0..448.0 for the same
tensors), so "read the scale as shaped and expand by ``out/s0`` rows and
``in/s1`` columns" is the correct layout.

PLE n-gram tables (``...ple_embedding.ngram_embedding.shard_{0..127}.weight``,
F8_E4M3 ``[2500012, 160]``, 320001536 rows over 128 shards)::

    row_value = mx.from_fp8(row_bytes, float32) * weight_scale

``weight_scale`` is the BF16 scalar ``ngram_embedding.weight_scale``
(bits 14673 = 1.9932e-4, identical to ``scale_bf16_bits`` in
``ple_fp8_manifest.json``). Rows are read with ``os.pread`` at
``manifest_offset + row * 160``; the manifest offsets were verified to equal
the safetensors data offset of each shard tensor, so a slice of three rows
costs 480 bytes rather than a 400 MB shard.

Residency classes
-----------------
``resident`` -- non-expert weights (embed_tokens, lm_head, attention,
PLE scalars): 9.4 GiB for the whole set, read on demand and kept decoded,
one tensor at a time.
``expert`` -- routed NVFP4 experts: decoded on demand, optionally cached in
an LRU holding N experts per layer.
``ple`` -- the 128 n-gram shard tables: serially row-addressed, cached by
row range under a byte budget.
``stream`` -- a 2-D tensor larger than the resident ceiling: addressable
only by row range, never whole.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlxcache_qwen4.shards import ShardCatalog, TensorEntry, array_from_bytes

__all__ = [
    "CheckpointLoader",
    "ResidencyStats",
    "classify",
    "DEFAULT_EXPERT_RESIDENT_N",
    "DEFAULT_PLE_CACHE_MB",
    "DEFAULT_RESIDENT_MAX_BYTES",
]

DEFAULT_EXPERT_RESIDENT_N = 0
DEFAULT_PLE_CACHE_MB = 512
# Ceiling for one whole resident tensor. The largest legitimate resident tensor
# is embed_tokens/lm_head at 248320*2560*2 = 1.27 GB; 4 GiB leaves headroom
# while still making "read the 400 MB PLE shard, let alone the 51.2 GB table"
# impossible through get().
DEFAULT_RESIDENT_MAX_BYTES = 4 << 30

_NVFP4_GROUP_SIZE = 16
_NVFP4_BITS = 4
# Strided slices are gathered row by row; the cap keeps a pathological slice
# from turning into a whole-table load.
_MAX_GATHER_ROWS = 1 << 16

_EXPERT_RE = re.compile(r"(?:^|\.)mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_PLE_SHARD_RE = re.compile(r"\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")


@dataclass(frozen=True)
class ResidencyStats:
    """Counters that make residency behaviour observable without RSS guessing.

    ``*_bytes_loaded`` count raw on-disk bytes read for that class (weight
    bytes plus the quantisation metadata read with them), so a test can assert
    that reading one expert transferred a few MB out of a 10 GB shard, and
    that a PLE row slice transferred only the requested rows.
    """

    expert_bytes_loaded: int
    expert_hits: int
    expert_misses: int
    ple_rows_loaded: int
    ple_bytes_loaded: int
    ple_hits: int
    ple_misses: int


class _ByteLRU:
    """LRU whose capacity is bytes, not entry count.

    Budgets are expressed in MB, so eviction has to be driven by the summed
    size of the decoded arrays; an entry larger than the whole budget is never
    inserted (it could not fit after any amount of eviction).

    Thread-safety: the sidecar serves requests on ThreadingHTTPServer worker
    threads and every worker shares one loader, so get/put race without the
    lock — two interleaved puts corrupt the `nbytes` invariant, after which
    the eviction loop never fires again and the cache grows without bound.

    Values are stored EVALUATED: a lazy graph built on one thread cannot be
    evaluated on another ("There is no Stream(gpu, 0) in current thread"), the
    same failure class `_materialize_weights` exists to prevent.
    """

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity = max(0, int(capacity_bytes))
        self._items: OrderedDict[Any, tuple[mx.array, int]] = OrderedDict()
        self._lock = threading.Lock()
        self.nbytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> mx.array | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return item[0]

    def put(self, key: Any, value: mx.array, nbytes: int) -> bool:
        if self.capacity == 0 or nbytes > self.capacity:
            return False
        mx.eval(value)
        with self._lock:
            existing = self._items.pop(key, None)
            if existing is not None:
                self.nbytes -= existing[1]
            self._items[key] = (value, nbytes)
            self.nbytes += nbytes
            while self.nbytes > self.capacity and len(self._items) > 1:
                _, (_, evicted) = self._items.popitem(last=False)
                self.nbytes -= evicted
            return True

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.nbytes = 0


def _decoded_nbytes(entry: TensorEntry) -> int:
    """Bytes a decoded 2-D tensor occupies in bf16 (the loader's output dtype).

    A packed NVFP4 weight stores two 4-bit codes per byte, so its logical width
    is twice the stored width; ignoring that halves the cache budget and lets a
    "hold N experts" setting hold 2N.
    """
    width = entry.shape[1] * 2 if entry.dtype == "U8" else entry.shape[1]
    return entry.shape[0] * width * 2


def classify(name: str) -> str:
    """Residency class of a checkpoint tensor name: ple, expert, or resident.

    The name alone decides this. Whether a resident tensor is instead STREAMED
    (read row-wise rather than decoded whole) depends on the loader instance's
    configured ceiling, so that decision lives in CheckpointLoader._classify
    and never here.
    """
    if _PLE_SHARD_RE.search(name):
        return "ple"
    if _EXPERT_RE.search(name):
        return "expert"
    return "resident"


class CheckpointLoader:
    """Serves full checkpoint tensor names, dequantized, under a memory budget."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        expert_resident_n: int | None = None,
        ple_cache_mb: float | None = None,
        resident_max_bytes: int | None = None,
    ) -> None:
        # Environment is read exactly once, here; no method ever looks at it.
        if expert_resident_n is None:
            expert_resident_n = int(
                os.environ.get("MLXCACHE_Q4_EXPERT_RESIDENT_N", DEFAULT_EXPERT_RESIDENT_N)
            )
        if ple_cache_mb is None:
            ple_cache_mb = float(os.environ.get("MLXCACHE_Q4_PLE_CACHE_MB", DEFAULT_PLE_CACHE_MB))
        self.expert_resident_n = max(0, int(expert_resident_n))
        self.ple_cache_mb = max(0.0, float(ple_cache_mb))
        self.resident_max_bytes = int(
            DEFAULT_RESIDENT_MAX_BYTES if resident_max_bytes is None else resident_max_bytes
        )

        self._catalog = ShardCatalog(checkpoint_dir)
        self._root = self._catalog.root
        self._resident: dict[str, mx.array] = {}
        # Capacity for "N experts per layer" is N times one whole expert's
        # decoded bytes, so the count knob and the byte-budgeted LRU agree.
        self._experts = _ByteLRU(self._expert_budget_bytes())
        self._ple = _ByteLRU(int(self.ple_cache_mb * (1 << 20)))
        self._ple_scales: dict[str, mx.array] = {}
        self._ple_manifest = self._load_manifest()

        self._expert_bytes_loaded = 0
        self._ple_rows_loaded = 0
        self._ple_bytes_loaded = 0
        self._closed = False

    # -- configuration derived once -------------------------------------
    def _expert_budget_bytes(self) -> int:
        """Bytes for N experts: N x one expert's decoded gate/up/down matrices.

        An expert is not one matrix. A routed expert is read and dequantized as
        three projections, and the block requests all three for every assigned
        expert, so budgeting N x the single largest matrix provisions only a
        third of an expert's footprint. With the budget that low the LRU is
        smaller than one token's routed set and never produces a hit.

        The per-expert total is derived from the catalog (the largest layer's
        gate+up+down sizes) rather than hardcoded, so the knob stays honest if
        the expert geometry changes.
        """
        if self.expert_resident_n == 0:
            return 0
        # Keyed by (module prefix, expert index), not layer number alone: two
        # families can both have an expert 0, and a bare layer number would sum
        # them into one budget.
        per_expert: dict[tuple[str, str], int] = {}
        for name in self._catalog.names():
            match = _EXPERT_RE.search(name)
            if not match or not name.endswith(".weight"):
                continue
            entry = self._catalog.entry(name)
            if entry.rank != 2:
                continue
            prefix = name.split(".mlp.experts.", 1)[0]
            key = (prefix, match.group(1))
            per_expert[key] = per_expert.get(key, 0) + _decoded_nbytes(entry)
        largest = max(per_expert.values(), default=0)
        return self.expert_resident_n * largest

    def _load_manifest(self) -> dict | None:
        path = self._root / "ple_fp8_manifest.json"
        if not path.is_file():
            return None
        with path.open("rb") as fh:
            manifest = json.load(fh)
        if manifest.get("layout") != "fp8_e4m3_safetensors_ranges":
            raise ValueError(f"{path}: unexpected layout {manifest.get('layout')!r}")
        return manifest

    # -- public surface --------------------------------------------------
    def get(self, name: str) -> mx.array:
        """Dequantized tensor, cached if it is resident-class."""
        entry = self._catalog.entry(name)
        cls = self._classify(entry)
        if cls == "ple":
            raise ValueError(
                f"{name!r} is a PLE n-gram table ({entry.nbytes / 2**30:.2f} GiB); "
                "use get_slice(name, rows) so only the requested rows are read"
            )
        if cls == "stream":
            raise ValueError(
                f"{name!r} decodes to {_decoded_nbytes(entry) / 2**30:.2f} GiB, over the "
                f"{self.resident_max_bytes / 2**30:.2f} GiB resident ceiling; "
                "use get_slice(name, rows)"
            )
        if cls == "resident":
            cached = self._resident.get(name)
            if cached is not None:
                return cached
            value = self._read_tensor(entry)
            self._resident[name] = value
            return value
        return self._get_expert(entry)

    def get_slice(self, name: str, rows: slice) -> mx.array:
        """Row range of a 2-D expert or PLE table, streaming only those rows."""
        entry = self._catalog.entry(name)
        cls = self._classify(entry)
        if cls == "resident":
            raise ValueError(
                f"{name!r} is classified 'resident'; get_slice streams expert and PLE "
                "tables, use get() for resident weights"
            )
        if entry.rank != 2:
            raise ValueError(f"{name!r} is rank {entry.rank}; get_slice needs a 2-D table")
        start, stop, step = rows.indices(entry.shape[0])
        if step <= 0:
            raise ValueError(f"{name!r}: only forward slices are supported, got step {step}")
        if stop <= start:
            raise ValueError(f"{name!r}: empty row range {rows!r}")
        if cls == "ple":
            return self._ple_slice(entry, start, stop, step)
        return self._expert_slice(entry, start, stop, step)

    def get_raw(self, name: str) -> tuple[mx.array, dict]:
        """(raw quantized array, header entry) for tests and inspection."""
        entry = self._catalog.entry(name)
        if entry.nbytes > self.resident_max_bytes:
            raise ValueError(
                f"{name!r} is {entry.nbytes / 2**30:.2f} GiB of raw bytes, over the "
                f"{self.resident_max_bytes / 2**30:.2f} GiB inspection ceiling"
            )
        header = {
            "dtype": entry.dtype,
            "shape": list(entry.shape),
            "data_offsets": [0, entry.nbytes],
            "shard": entry.shard,
        }
        return self._catalog.array(entry), header

    def stats(self) -> ResidencyStats:
        return ResidencyStats(
            expert_bytes_loaded=self._expert_bytes_loaded,
            expert_hits=self._experts.hits,
            expert_misses=self._experts.misses,
            ple_rows_loaded=self._ple_rows_loaded,
            ple_bytes_loaded=self._ple_bytes_loaded,
            ple_hits=self._ple.hits,
            ple_misses=self._ple.misses,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._resident.clear()
        self._experts.clear()
        self._ple.clear()
        self._catalog.close()

    # -- internals -------------------------------------------------------
    def _classify(self, entry: TensorEntry) -> str:
        cls = classify(entry.name)
        if (
            cls == "resident"
            and entry.rank == 2
            and _decoded_nbytes(entry) > self.resident_max_bytes
        ):
            return "stream"
        return cls

    def _read_tensor(self, entry: TensorEntry) -> mx.array:
        if entry.nbytes > self.resident_max_bytes:
            raise ValueError(
                f"{entry.name!r} is {entry.nbytes / 2**30:.2f} GiB of raw bytes, over the "
                f"{self.resident_max_bytes / 2**30:.2f} GiB resident ceiling"
            )
        if _EXPERT_RE.search(entry.name):
            self._expert_bytes_loaded += entry.nbytes
        return self._catalog.array(entry)

    def _is_packed_weight(self, entry: TensorEntry) -> bool:
        """True for a quantized payload tensor that needs its scale metadata.

        The test is structural rather than dtype-based: a 2-D U8/F8 tensor named
        ``*.weight`` counts only if its companion scale table exists. That keeps
        ``weight_scale`` (itself a 2-D F8 tensor) from being mistaken for
        payload and recursed into.
        """
        if entry.rank != 2 or not entry.name.endswith(".weight"):
            return False
        if entry.dtype == "U8":
            return self._catalog.has(entry.name.replace(".weight", ".weight_scale"))
        if entry.dtype == "F8_E4M3":
            return self._catalog.has(entry.name.replace(".weight", ".weight_scale_inv"))
        return False

    def _account_expert(self, nbytes: int) -> None:
        self._expert_bytes_loaded += nbytes

    # experts ------------------------------------------------------------
    def _get_expert(self, entry: TensorEntry) -> mx.array:
        cached = self._experts.get(entry.name)
        if cached is not None:
            return cached
        if not self._is_packed_weight(entry):
            return self._read_tensor(entry)
        value, loaded = self._decode_rows(entry, 0, entry.shape[0], 1)
        self._account_expert(loaded)
        self._experts.put(entry.name, value, value.nbytes)
        return value

    def _expert_slice(self, entry: TensorEntry, start: int, stop: int, step: int) -> mx.array:
        if not self._is_packed_weight(entry):
            return self._plain_rows(entry, start, stop, step)
        value, loaded = self._decode_rows(entry, start, stop, step)
        self._account_expert(loaded)
        return value

    def _decode_rows(
        self, entry: TensorEntry, start: int, stop: int, step: int
    ) -> tuple[mx.array, int]:
        """Dequantize expert rows [start, stop), returning (value, bytes read)."""
        if entry.dtype == "U8":
            return self._decode_nvfp4(entry, start, stop, step)
        if entry.dtype == "F8_E4M3":
            return self._decode_fp8(entry, start, stop, step)
        raise ValueError(f"{entry.name!r}: {entry.dtype} is not a quantized expert weight")

    def _decode_nvfp4(
        self, entry: TensorEntry, start: int, stop: int, step: int
    ) -> tuple[mx.array, int]:
        match = _EXPERT_RE.search(entry.name)
        scale_name = entry.name.replace(".weight", ".weight_scale")
        scale_entry = self._catalog.entry(scale_name)
        if scale_entry.shape[0] != entry.shape[0]:
            raise ValueError(
                f"{scale_name!r} has {scale_entry.shape[0]} rows for "
                f"{entry.shape[0]} weight rows; nvfp4 scales are per row"
            )
        weight_rows, wbytes = self._rows(entry, start, stop, step)
        scale_rows, sbytes = self._rows(scale_entry, start, stop, step)
        global_scale, gbytes = self._global_scale(
            entry.name,
            match.group(2) if match else None,
            match.group(1) if match else None,
            _LAYER_RE.search(entry.name).group(1) if _LAYER_RE.search(entry.name) else None,
        )
        width = entry.shape[1] * 2
        if weight_rows.shape[1] * 2 != width:
            raise ValueError(f"{entry.name!r}: packed row is not half the logical width")
        packed = weight_rows.view(mx.uint32).reshape(weight_rows.shape[0], width // 8)
        value = mx.dequantize(
            packed,
            scales=scale_rows,
            group_size=_NVFP4_GROUP_SIZE,
            bits=_NVFP4_BITS,
            mode="nvfp4",
            global_scale=global_scale,
            dtype=mx.bfloat16,
        )
        return value, wbytes + sbytes + gbytes

    def _global_scale(
        self, name: str, proj: str | None, expert: str | None, layer: str | None
    ) -> tuple[mx.array, int]:
        """Per-expert global scale scalar and the bytes spent reading it.

        Raises when a routed-expert name cannot be resolved to its table: the
        dequantization is catastrophically wrong without the per-expert scale
        (measured rms 204.0 vs 0.0135 with and without it, a ~20,500x error),
        and a checkpoint revision that renames or drops the table must fail
        loudly at load, not serve garbage routed experts."""
        if proj is None or expert is None or layer is None:
            raise ValueError(
                f"{name!r}: routed expert name does not carry layer/projection/expert; "
                "cannot locate its global_scale table"
            )
        table_name = f"model.language_model.layers.{layer}.mlp.switch_mlp.{proj}.global_scale"
        if not self._catalog.has(table_name):
            raise ValueError(f"{name!r}: per-expert global_scale table {table_name!r} is absent")
        table = self._catalog.entry(table_name)
        index = int(expert)
        if table.rank != 1 or index >= table.shape[0]:
            raise ValueError(
                f"{table_name!r} is {list(table.shape)} and cannot supply expert {index}"
            )
        # Four bytes, not the 2 KB table: only this expert's scale is needed.
        raw = self._catalog.read(table, index * 4, 4)
        return array_from_bytes(raw, "F32", ()), 4

    def _decode_fp8(
        self, entry: TensorEntry, start: int, stop: int, step: int
    ) -> tuple[mx.array, int]:
        scale_name = entry.name.replace(".weight", ".weight_scale_inv")
        scale_entry = self._catalog.entry(scale_name)
        block_rows, block_cols = self._block_shape(scale_entry, entry)
        weight_rows, wbytes = self._rows(entry, start, stop, step)
        block_ids = [row // block_rows for row in range(start, stop, step)]
        scale_rows, sbytes = self._scale_rows(scale_entry, block_ids)
        values = mx.from_fp8(weight_rows, dtype=mx.float32)
        # One scale covers block_rows weight rows and block_cols columns; widen
        # it across the columns, one row per requested weight row.
        scales = mx.repeat(scale_rows.astype(mx.float32), block_cols, axis=1)
        if scales.shape[0] != values.shape[0] or scales.shape[1] != values.shape[1]:
            raise ValueError(
                f"{scale_name!r}: expanded scales {list(scales.shape)} do not match "
                f"weights {list(values.shape)} for rows {start}..{stop}"
            )
        return (values * scales).astype(mx.bfloat16), wbytes + sbytes

    @staticmethod
    def _block_shape(scale_entry: TensorEntry, weight_entry: TensorEntry) -> tuple[int, int]:
        out, width = weight_entry.shape
        if scale_entry.rank != 2:
            raise ValueError(f"{scale_entry.name!r} is not a 2-D block-scale table")
        s_out, s_width = scale_entry.shape
        if out % s_out or width % s_width:
            raise ValueError(
                f"{scale_entry.name!r}{list(scale_entry.shape)} does not tile "
                f"{weight_entry.name!r}{list(weight_entry.shape)}"
            )
        return out // s_out, width // s_width

    def _scale_rows(self, scale_entry: TensorEntry, block_ids: list[int]) -> tuple[mx.array, int]:
        """One block-scale row per requested weight row, deduplicated on disk."""
        unique = sorted(set(block_ids))
        if len(unique) == 1:
            only = unique[0]
            buf = self._catalog.read(
                scale_entry, only * scale_entry.row_nbytes, scale_entry.row_nbytes
            )
            one = array_from_bytes(buf, scale_entry.dtype, (1, scale_entry.shape[1]))
            # Every requested row lands in the same block: hand back one row per
            # weight row so the caller's elementwise multiply stays shape-exact.
            return mx.stack([one[0]] * len(block_ids)), len(buf)
        lo, hi = unique[0], unique[-1]
        contiguous = hi - lo + 1 == len(unique)
        if contiguous:
            buf = self._catalog.read(
                scale_entry, lo * scale_entry.row_nbytes, (hi - lo + 1) * scale_entry.row_nbytes
            )
            table = array_from_bytes(buf, scale_entry.dtype, (hi - lo + 1, scale_entry.shape[1]))
            nbytes = len(buf)
        else:
            chunks = [
                self._catalog.read(
                    scale_entry, row * scale_entry.row_nbytes, scale_entry.row_nbytes
                )
                for row in unique
            ]
            table = array_from_bytes(
                b"".join(chunks), scale_entry.dtype, (len(unique), scale_entry.shape[1])
            )
            nbytes = sum(len(chunk) for chunk in chunks)
        # O(1) position lookup: a linear `unique.index` scan per requested row
        # made dequantizing a 2560-row expert matrix a rows x blocks Python loop.
        offset = {block_id: i for i, block_id in enumerate(unique)}
        positions = [offset[block_id] for block_id in block_ids]
        return mx.stack([table[position] for position in positions]), nbytes

    # raw rows -----------------------------------------------------------
    def _rows(self, entry: TensorEntry, start: int, stop: int, step: int) -> tuple[mx.array, int]:
        """Raw rows [start, stop) of one 2-D tensor, touching only those bytes."""
        row_bytes = entry.row_nbytes
        if step == 1:
            buf = self._catalog.read(entry, start * row_bytes, (stop - start) * row_bytes)
            value = array_from_bytes(buf, entry.dtype, (stop - start, entry.shape[1]))
            return value, len(buf)
        if stop - start > _MAX_GATHER_ROWS:
            raise ValueError(
                f"{entry.name!r}: strided slice of {stop - start} rows exceeds the "
                f"{_MAX_GATHER_ROWS}-row per-row read ceiling; read a contiguous range"
            )
        chunks = [
            self._catalog.read(entry, row * row_bytes, row_bytes)
            for row in range(start, stop, step)
        ]
        value = array_from_bytes(b"".join(chunks), entry.dtype, (len(chunks), entry.shape[1]))
        return value, sum(len(chunk) for chunk in chunks)

    def _plain_rows(self, entry: TensorEntry, start: int, stop: int, step: int) -> mx.array:
        value, _ = self._rows(entry, start, stop, step)
        return value

    # PLE ---------------------------------------------------------------
    def _ple_shard_info(self, entry: TensorEntry) -> tuple[int, int]:
        """(shard index, row count) after cross-checking the manifest."""
        match = _PLE_SHARD_RE.search(entry.name)
        if match is None:
            raise ValueError(f"{entry.name!r} is not a PLE n-gram shard table")
        index = int(match.group(1))
        manifest = self._ple_manifest
        if manifest is None:
            return index, entry.shape[0]
        shards = manifest["shards"]
        if index >= len(shards):
            raise ValueError(f"{entry.name!r}: shard {index} missing from the PLE manifest")
        info = shards[index]
        row_count = int(info["row_count"])
        if (row_count, int(manifest["row_width"])) != entry.shape:
            raise ValueError(
                f"{entry.name!r}: manifest says ({row_count}, {manifest['row_width']}), "
                f"header says {list(entry.shape)}"
            )
        if Path(info["file"]).name != Path(entry.shard).name:
            raise ValueError(f"{entry.name!r}: manifest file {info['file']} != shard {entry.shard}")
        if int(info["offset"]) != entry.abs_start:
            raise ValueError(
                f"{entry.name!r}: manifest offset {info['offset']} != header offset "
                f"{entry.abs_start}; refusing to trust either"
            )
        return index, row_count

    def _ple_scale(self, entry: TensorEntry) -> mx.array:
        prefix, _, _ = entry.name.partition(".shard_")
        if not prefix.endswith(".ngram_embedding"):
            raise ValueError(f"{entry.name!r} has no sibling ngram_embedding weight_scale")
        cached = self._ple_scales.get(prefix)
        if cached is not None:
            return cached
        scale_entry = self._catalog.entry(prefix + ".weight_scale")
        value = self._catalog.array(scale_entry).astype(mx.float32)
        # One 4-byte scalar per PLE layer, decoded once for the loader's life:
        # re-reading it on every row-cache miss added a catalog lookup, a file
        # read, and a host-to-device copy to each new n-gram run.
        self._ple_scales[prefix] = value
        return value

    def _ple_slice(self, entry: TensorEntry, start: int, stop: int, step: int) -> mx.array:
        index, row_count = self._ple_shard_info(entry)
        if stop > row_count:
            raise ValueError(
                f"{entry.name!r} shard {index} has {row_count} rows, {start}..{stop} requested"
            )
        # Keyed by the TENSOR NAME, not the bare shard index: shard tables are
        # per-PLE-layer, so two layers' shard_0 would otherwise alias in the
        # LRU and layer B would silently receive layer A's decoded rows.
        key = (entry.name, start, stop, step)
        cached = self._ple.get(key)
        if cached is not None:
            return cached
        rows, nbytes = self._rows(entry, start, stop, step)
        value = (mx.from_fp8(rows, dtype=mx.float32) * self._ple_scale(entry)).astype(mx.bfloat16)
        self._ple_rows_loaded += len(range(start, stop, step))
        self._ple_bytes_loaded += nbytes
        self._ple.put(key, value, value.nbytes)
        return value
