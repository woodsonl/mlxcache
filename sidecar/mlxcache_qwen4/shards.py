"""Byte-range access to the sharded safetensors checkpoint.

Why this module reads bytes itself instead of letting safetensors decode:
this environment's numpy has no ``bfloat16`` or ``float8_e4m3fn`` dtypes, so
every safetensors tensor path raises for the BF16 and F8_E4M3 tensors that
make up almost the whole checkpoint (verified:
``safe_open(path, framework="mlx").get_tensor("...shared_expert.gate_proj.weight")``
-> ``TypeError: data type 'bfloat16' not understood``). ``safe_open`` is still
the catalog authority here -- it parses and validates the header, and our own
header parse is checked against its reported shapes and keys -- but tensor
bytes come from ``os.pread`` over the exact ``[start, stop)`` range so that
"read nothing you did not ask for" is a property of this code rather than a
promise about mmap behaviour.

``safe_open`` is opened with ``backend="pread"`` and ``device="cpu"``: the
checkpoint is 132.6 GB on a 48 GB machine, so nothing may be mapped.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

# safetensors dtype -> element size in bytes. The file format stores raw
# little-endian bytes tagged with these names; BF16/F8_E4M3 are two of the
# names numpy cannot turn into an array, which is why we decode them ourselves.
ITEMSIZE: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

# dtypes whose numpy counterpart exists, so they can be built without a view.
_NUMPY_DTYPES: dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "U16": np.dtype(np.uint16),
    "I16": np.dtype(np.int16),
    "F16": np.dtype(np.float16),
    "U32": np.dtype(np.uint32),
    "I32": np.dtype(np.int32),
    "F32": np.dtype(np.float32),
    "U64": np.dtype(np.uint64),
    "I64": np.dtype(np.int64),
    "F64": np.dtype(np.float64),
}


def shape_elements(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def array_from_bytes(buf: bytes, dtype: str, shape: tuple[int, ...]) -> mx.array:
    """Decode raw little-endian safetensors bytes into an mlx array.

    ``F8_E4M3`` stays as raw ``uint8`` payload bytes: mlx has no fp8 dtype
    object, and ``mx.from_fp8``/``mx.dequantize`` both take the raw byte
    payload. Callers that need values convert explicitly.
    """
    try:
        itemsize = ITEMSIZE[dtype]
    except KeyError:  # pragma: no cover - only reachable for a new format tag
        raise ValueError(f"unsupported safetensors dtype {dtype!r}") from None
    expected = shape_elements(shape) * itemsize
    if len(buf) != expected:
        raise ValueError(f"short read: {len(buf)} bytes for {dtype}{list(shape)}, want {expected}")

    if dtype == "BF16":
        # uint16 view then dtype reinterpretation; mlx copies, so the temporary
        # numpy buffer is free to die here.
        arr = mx.array(np.frombuffer(buf, dtype=np.uint16)).view(mx.bfloat16)
    elif dtype in _NUMPY_DTYPES:
        arr = mx.array(np.frombuffer(buf, dtype=_NUMPY_DTYPES[dtype]))
    elif dtype in ("F8_E4M3", "F8_E5M2"):
        arr = mx.array(np.frombuffer(buf, dtype=np.uint8))
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    return arr.reshape(shape)


@dataclass(frozen=True)
class TensorEntry:
    """One tensor's location: which shard, its dtype/shape, absolute byte range."""

    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    abs_start: int
    abs_stop: int

    @property
    def nbytes(self) -> int:
        return self.abs_stop - self.abs_start

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def row_nbytes(self) -> int:
        """Bytes per row for a 2-D tensor (0 for other ranks)."""
        if self.rank != 2:
            return 0
        return self.shape[1] * ITEMSIZE[self.dtype]


class ShardCatalog:
    """Resolves tensor names to shards and reads exact byte ranges."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.root = Path(checkpoint_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {self.root}")
        self._index = self._load_index()
        self._headers: dict[str, tuple[dict, int]] = {}
        self._entries: dict[str, TensorEntry] = {}
        self._specs: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {}
        self._handles: dict[str, object] = {}
        self.bytes_read = 0

    # -- catalog ---------------------------------------------------------
    def _load_index(self) -> dict[str, str]:
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            with index_path.open("rb") as fh:
                payload = json.load(fh)
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"{index_path} has no weight_map")
            return dict(weight_map)
        single = self.root / "model.safetensors"
        if single.is_file():
            header, _ = self._read_header("model.safetensors")
            return {
                name: "model.safetensors"
                for name, meta in header.items()
                if name != "__metadata__" and isinstance(meta, dict)
            }
        raise FileNotFoundError(f"no model.safetensors(.index.json) in {self.root}")

    def _read_header(self, shard: str) -> tuple[dict, int]:
        """Parse the safetensors header: 8-byte LE length, then UTF-8 JSON."""
        path = self.root / shard
        with open(path, "rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                raise ValueError(f"{path} is not a safetensors file (truncated header length)")
            (header_len,) = struct.unpack("<Q", raw_len)
            header = json.loads(fh.read(header_len))
        return header, 8 + header_len

    def _header_for(self, shard: str) -> tuple[dict, int]:
        cached = self._headers.get(shard)
        if cached is None:
            cached = self._read_header(shard)
            self._headers[shard] = cached
        return cached

    def _specs_for(self, shard: str) -> dict[str, tuple[tuple[int, ...], str]]:
        """name -> (shape, dtype) as safetensors itself reports them.

        Built once per shard from the parse-only handle: ``get_slice`` does not
        read tensor data, so this is a header walk, not a 10 GB load. Doing it
        per shard rather than per tensor matters -- a shard holds ~40k tensors
        and re-asking for each one made construction take minutes.
        """
        cached = self._specs.get(shard)
        if cached is None:
            # backend="pread": never mmap a 10 GB shard on a 48 GB machine.
            with safe_open(
                str(self.root / shard), framework="mlx", device="cpu", backend="pread"
            ) as handle:
                cached = {
                    name: (
                        tuple(handle.get_slice(name).get_shape()),
                        handle.get_slice(name).get_dtype(),
                    )
                    for name in handle.keys()  # noqa: SIM118 -- keys() is correct here
                }
            self._specs[shard] = cached
        return cached

    def names(self) -> list[str]:
        return list(self._index)

    def shard_names(self) -> list[str]:
        return sorted(set(self._index.values()))

    def has(self, name: str) -> bool:
        return name in self._index

    def entry(self, name: str) -> TensorEntry:
        cached = self._entries.get(name)
        if cached is not None:
            return cached
        try:
            shard = self._index[name]
        except KeyError:
            raise KeyError(self._unknown_message(name)) from None
        header, base = self._header_for(shard)
        meta = header.get(name)
        if not isinstance(meta, dict):
            # The index and the shard header disagree: treat it as unknown
            # rather than blowing up later with a struct error mid-read.
            raise KeyError(self._unknown_message(name))
        start, stop = meta["data_offsets"]
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        itemsize = ITEMSIZE.get(dtype)
        if itemsize is None:
            raise ValueError(f"{name}: unsupported dtype {dtype!r}")
        expected = shape_elements(shape) * itemsize
        if stop - start != expected:
            raise ValueError(
                f"{name}: header byte range {start}..{stop} does not match {dtype}{list(shape)}"
            )
        spec = self._specs_for(shard).get(name)
        if spec is not None and spec != (shape, dtype):
            raise ValueError(
                f"{name}: raw header says {dtype}{list(shape)}, "
                f"safetensors says {spec[1]}{list(spec[0])}"
            )
        entry = TensorEntry(
            name=name,
            shard=shard,
            dtype=dtype,
            shape=shape,
            abs_start=base + start,
            abs_stop=base + stop,
        )
        self._entries[name] = entry
        return entry

    def _unknown_message(self, name: str) -> str:
        return (
            f"{name!r} is not a tensor in {self.root} "
            f"(catalog holds {len(self._index)} names); refusing to guess"
        )

    # -- bytes -----------------------------------------------------------
    def read(self, entry: TensorEntry, start: int = 0, nbytes: int | None = None) -> bytes:
        """Read ``[start, start+nbytes)`` relative to the tensor's own bytes."""
        if nbytes is None:
            nbytes = entry.nbytes - start
        if start < 0 or nbytes < 0 or start + nbytes > entry.nbytes:
            raise ValueError(
                f"{entry.name}: range {start}..{start + nbytes} outside tensor "
                f"of {entry.nbytes} bytes"
            )
        path = self.root / entry.shard
        fd = os.open(path, os.O_RDONLY)
        try:
            buf = os.pread(fd, nbytes, entry.abs_start + start)
        finally:
            os.close(fd)
        if len(buf) != nbytes:
            raise ValueError(f"{entry.name}: pread returned {len(buf)} of {nbytes} bytes")
        self.bytes_read += len(buf)
        return buf

    def array(self, entry: TensorEntry) -> mx.array:
        return array_from_bytes(self.read(entry), entry.dtype, entry.shape)

    def array_range(
        self, entry: TensorEntry, start: int, nbytes: int, shape: tuple[int, ...]
    ) -> mx.array:
        return array_from_bytes(self.read(entry, start, nbytes), entry.dtype, shape)

    def open_shards(self) -> dict[str, list[str]]:
        """Shard -> tensor names, in shard order (used by PLE shard discovery)."""
        out: dict[str, list[str]] = {}
        for name, shard in self._index.items():
            out.setdefault(shard, []).append(name)
        return out

    def close(self) -> None:
        self._specs.clear()
        self._entries.clear()
        self._headers.clear()
