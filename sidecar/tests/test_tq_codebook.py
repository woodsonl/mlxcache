"""Lloyd-Max codebook tests — hermetic (no mlx, no numpy).

The validation anchors are the TurboQuant paper's own numbers (arXiv
2504.19874 §3.1, Theorem 1), which the module derives independently from
the exact Beta marginal. If these tests fail after an edit, the quantizer
drifted from the paper — the failure is the point.
"""

from __future__ import annotations

import math

import pytest
from mlxcache_sidecar.tq.codebook import lloyd_max_codebook


@pytest.mark.parametrize("bits,anchor", [(1, 0.36), (2, 0.117), (3, 0.03), (4, 0.009)])
def test_mse_matches_paper_anchors(bits, anchor):
    # The paper rounds its own solve (its b=3/b=4 anchors sit below Max's
    # Gaussian tables because the exact Beta has lighter tails); 15% bands
    # are tight enough to catch a wrong density or a broken centroid update
    # while tolerating the paper's rounding.
    _, _, mse = lloyd_max_codebook(128, bits)
    assert mse == pytest.approx(anchor, rel=0.15)


def test_b1_centroids_match_closed_form():
    levels, _, _ = lloyd_max_codebook(128, 1)
    assert levels[1] == pytest.approx(math.sqrt(2 / math.pi), rel=0.01)
    assert levels[0] == pytest.approx(-math.sqrt(2 / math.pi), rel=0.01)


def test_b2_centroids_match_paper():
    levels, _, _ = lloyd_max_codebook(128, 2)
    magnitudes = sorted({round(abs(v), 3) for v in levels})
    assert magnitudes == pytest.approx([0.453, 1.51], abs=0.01)


def test_distortion_scales_like_one_over_4b():
    mses = [lloyd_max_codebook(128, b)[2] for b in (2, 3, 4)]
    ratios = [mses[i] / mses[i + 1] for i in range(2)]
    # Theorem 3's 1/4^b scaling: each extra bit divides distortion ~4x.
    assert ratios[0] == pytest.approx(4, rel=0.25)
    assert ratios[1] == pytest.approx(4, rel=0.25)


def test_fixed_point_optimality():
    """At the Max-Lloyd fixed point, boundaries are the exact midpoints of
    adjacent centroids (the property `encode_keys`'s bucketing relies on)."""
    levels, boundaries, _ = lloyd_max_codebook(64, 3)
    assert len(levels) == 8 and len(boundaries) == 7
    for c in range(7):
        assert boundaries[c] == pytest.approx((levels[c] + levels[c + 1]) / 2, abs=1e-9)
    assert list(levels) == sorted(levels)


def test_levels_are_symmetric():
    levels, _, _ = lloyd_max_codebook(256, 4)
    for lo, hi in zip(levels, reversed(levels), strict=True):
        assert lo == pytest.approx(-hi, abs=1e-9)


def test_deterministic_and_cached():
    a = lloyd_max_codebook(128, 3)
    b = lloyd_max_codebook(128, 3)
    assert a is b  # lru_cache: one solve per (d, bits) per process
    assert a[0] == lloyd_max_codebook(128, 3)[0]


@pytest.mark.parametrize("d,bits", [(1, 3), (128, 0), (128, 7), (128, -1)])
def test_rejects_bad_arguments(d, bits):
    with pytest.raises(ValueError):
        lloyd_max_codebook(d, bits)
