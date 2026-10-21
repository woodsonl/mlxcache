"""Lloyd-Max codebooks for the TurboQuant key quantizer.

The quantizer is specified by the TurboQuant paper (arXiv:2504.19874 §3.1,
Lemma 1): a random orthogonal rotation maps any unit vector's coordinates to
the marginal

    f_X(x) = Γ(d/2) / (√π·Γ((d-1)/2)) · (1 - x²)^((d-3)/2),  x ∈ [-1, 1],

and the optimal b-bit scalar quantizer for that density is the fixed point of
the Max-Lloyd iteration (continuous 1-D k-means). This module DERIVES the
codebooks from that density — nothing is copied from any third-party
implementation; the validation anchors below come from the paper's own text.

Working variable: u = x·√d. The density of u is g(u) ∝ (1 - u²/d)^((d-3)/2)
on [-√d, √d), which concentrates like N(0, 1). Solving in u-units keeps the
grid O(1) wide for every head_dim and makes the paper's Gaussian-approximation
anchors directly comparable:

- b=1 centroids ±√(2/π) ≈ ±0.7979 (paper §3.1)
- b=2 centroids {±0.453, ±1.51} (paper §3.1)
- per-coordinate MSE (u-units) ≈ 0.36 / 0.117 / 0.03 / 0.009 for b = 1..4
  (paper Theorem 1; the paper's own rounding of the exact-Beta solve)

Integration: a uniform midpoint grid over [-8, 8] (tail mass beyond 8/√d in
x-units is < 1e-14 for every d ≥ 2) with prefix sums of (g, u·g, u²·g) built
once per head_dim, so every Max-Lloyd iteration is O(k) instead of O(N).
Pure `math` on purpose: zero dependencies, so the hermetic (no-mlx, no-numpy)
test suite can validate the quantizer math directly.
"""

from __future__ import annotations

import bisect
import functools
import math

_GRID_N = 40_001
_GRID_U = 8.0


@functools.lru_cache(maxsize=1)
def _grid_us() -> tuple[float, ...]:
    du = 2 * _GRID_U / _GRID_N
    return tuple(-_GRID_U + (i + 0.5) * du for i in range(_GRID_N))


@functools.lru_cache(maxsize=8)
def _prefix_sums(d: int) -> tuple[list[float], list[float], list[float]]:
    """Prefix sums of (g, u·g, u²·g)·du over the grid for head_dim d."""
    us = _grid_us()
    du = 2 * _GRID_U / _GRID_N
    exponent = (d - 3) / 2.0
    inv_d = 1.0 / d
    p0 = [0.0] * (_GRID_N + 1)
    p1 = [0.0] * (_GRID_N + 1)
    p2 = [0.0] * (_GRID_N + 1)
    for i, u in enumerate(us):
        # (1 - u²/d)^((d-3)/2) in exp-log form: the raw power underflows to
        # 0.0 deep in the tails where the density is still representable.
        t = 1.0 - u * u * inv_d
        g = math.exp(exponent * math.log(t)) if t > 0.0 else 0.0
        gu = g * du
        p0[i + 1] = p0[i] + gu
        p1[i + 1] = p1[i] + gu * u
        p2[i + 1] = p2[i] + gu * u * u
    return p0, p1, p2


@functools.lru_cache(maxsize=64)
def lloyd_max_codebook(
    d: int, bits: int, *, iters: int = 300
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """(levels, interior_boundaries, per-coordinate MSE) in u-units.

    Deterministic: equal-mass cell initialization from the CDF, then
    Max-Lloyd (centroids ← cell means, edges ← midpoints of adjacent
    centroids) until the grid-quantized edges stop moving. Levels ascend;
    `boundaries` holds the k-1 EXACT midpoints of adjacent levels — the
    encode-side thresholds. The returned MSE is E[(u − ĉ)²] under g with
    the grid-integrated cells, comparable to the paper's Theorem-1 anchors
    (0.36 / 0.117 / 0.03 / 0.009 for b = 1..4).
    """
    if d < 2:
        raise ValueError("head_dim must be >= 2")
    if not 1 <= bits <= 6:
        raise ValueError("bits must be in 1..6")
    k = 1 << bits
    p0, p1, p2 = _prefix_sums(d)
    us = _grid_us()
    total = p0[_GRID_N]

    def cell_stats(lo: int, hi: int) -> tuple[float, float, float]:
        return p0[hi] - p0[lo], p1[hi] - p1[lo], p2[hi] - p2[lo]

    def grid_index(u: float) -> int:
        """First grid index whose midpoint is >= u, clamped to [1, N-1]."""
        return min(max(bisect.bisect_left(us, u), 1), _GRID_N - 1)

    # Equal-mass initialization: interior edges at the k-1 CDF quantiles.
    edges = [bisect.bisect_left(p0, total * j / k) for j in range(1, k)]
    edges = [min(max(e, 1), _GRID_N - 1) for e in edges]

    def centroids_from_edges() -> list[float]:
        cells = [0, *edges, _GRID_N]
        out = []
        for c in range(k):
            m0, m1, _ = cell_stats(cells[c], cells[c + 1])
            out.append(m1 / m0 if m0 > 0.0 else us[min(cells[c], _GRID_N - 1)])
        return out

    for _ in range(iters):
        levels = centroids_from_edges()
        new_edges = [grid_index((levels[c] + levels[c + 1]) / 2.0) for c in range(k - 1)]
        if new_edges == edges:
            break
        edges = new_edges

    levels = centroids_from_edges()
    # The density is symmetric, so the exact fixed point is too; the grid-
    # snapped edges leave O(du) asymmetry. Symmetrize the solved levels and
    # derive boundaries from them, keeping the returned pair self-consistent.
    for c in range(k // 2):
        m = (levels[c] - levels[k - 1 - c]) / 2.0
        levels[c], levels[k - 1 - c] = m, -m
    boundaries = tuple((levels[c] + levels[c + 1]) / 2.0 for c in range(k - 1))

    cells = [0, *edges, _GRID_N]
    mse = 0.0
    for c in range(k):
        m0, m1, m2 = cell_stats(cells[c], cells[c + 1])
        mse += m2 - 2.0 * levels[c] * m1 + levels[c] * levels[c] * m0
    mse /= total
    return tuple(levels), boundaries, mse
