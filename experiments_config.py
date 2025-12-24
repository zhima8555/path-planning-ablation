"""Centralized, reproducible evaluation configuration.

This module defines shared constants and helper utilities that are imported by
multiple evaluation scripts to ensure experiments are comparable and
reproducible.
"""

from __future__ import annotations

from typing import List


# Map categories used across ablation evaluations
MAP_TYPES = ["simple", "complex", "concave", "narrow"]

# Static evaluation: number of start-goal pairs per map type (K) and number of
# repeated runs per pair (N).
STATIC_K = 20
STATIC_N = 1

# Dynamic evaluation: number of scenarios per map type (K) and number of
# repeated runs per scenario (N).
DYNAMIC_K = 20
DYNAMIC_N = 10

# Maximum planner steps
MAX_STEPS_STATIC = 500
MAX_STEPS_DYNAMIC = 800


def get_seeds(base: int, n: int) -> List[int]:
    """Return a deterministic list of integer seeds.

    Args:
        base: Base seed.
        n: Number of seeds.

    Returns:
        A list of length n: [base, base+1, ..., base+n-1].
    """

    return [base + i for i in range(n)]
