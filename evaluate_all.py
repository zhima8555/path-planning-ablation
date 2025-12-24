"""Evaluate all methods across all predefined map types.

This script is intended for reproducible ablation evaluation.

Changes vs. previous version:
- Uses shared constants from experiments_config.py.
- Uses deterministic per-map fixed start-goal pairs (K=20) generated from seeds.
- Reports success_count, collision_count, collision_rate in addition to
  success_rate and path_length.
- Saves outputs to both CSV and TXT in the repository root.

Note: This script assumes existing project utilities for creating maps,
      sampling valid start/goal states, and running planners.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from experiments_config import (
    MAP_TYPES,
    STATIC_K,
    STATIC_N,
    MAX_STEPS_STATIC,
    get_seeds,
)


# ------------------------------ Helpers ------------------------------------

@dataclass
class Metrics:
    success_count: int = 0
    collision_count: int = 0
    total_count: int = 0
    sum_path_length: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count else 0.0

    @property
    def collision_rate(self) -> float:
        return self.collision_count / self.total_count if self.total_count else 0.0

    @property
    def avg_path_length(self) -> float:
        # Only average path length over successful episodes to match common
        # reporting; if project historically averaged differently, adjust here.
        return self.sum_path_length / self.success_count if self.success_count else 0.0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def extract_episode_outcome(result: Any) -> Tuple[bool, bool, float]:
    """Best-effort extraction to keep backward compatibility.

    Returns:
        (success, collision, path_length)

    The repo has had different return structures across planners/versions. This
    function tries several common patterns:
    - dict with keys: success, collided/collision, path_length/length
    - tuple/list: (success, path, info) etc.
    """

    # dict-like
    if isinstance(result, dict):
        success = bool(result.get("success", False))
        collision = bool(
            result.get("collision", result.get("collided", result.get("is_collision", False)))
        )
        path_len = _safe_float(
            result.get(
                "path_length",
                result.get("length", result.get("path_len", result.get("traj_length", 0.0))),
            ),
            0.0,
        )
        return success, collision, path_len

    # tuple/list
    if isinstance(result, (tuple, list)) and len(result) > 0:
        # If first element looks like success boolean
        success = bool(result[0])
        collision = False
        path_len = 0.0
        # try to find info dict
        info = None
        for item in result[1:]:
            if isinstance(item, dict):
                info = item
                break
        if info is not None:
            collision = bool(info.get("collision", info.get("collided", False)))
            path_len = _safe_float(info.get("path_length", info.get("length", 0.0)), 0.0)
        return success, collision, path_len

    return False, False, 0.0


# -------------------------- Fixed pair generation ---------------------------

def generate_fixed_start_goal_pairs(map_type: str, k: int) -> List[Tuple[Any, Any]]:
    """Generate deterministic fixed start-goal pairs for a map type.

    This function relies on existing project utilities:
    - make_map(map_type, seed=...)
    - sample_start_goal(world, rng)

    If these utilities differ in your codebase, update the imports and logic
    accordingly.
    """

    # Local imports to avoid import-time side effects.
    # Expected to exist in the project.
    from utils.maps import make_map  # type: ignore
    from utils.sampling import sample_start_goal  # type: ignore

    # Use a stable base seed per map type.
    base = 10_000 + MAP_TYPES.index(map_type) * 1_000
    pair_seeds = get_seeds(base, k)

    pairs: List[Tuple[Any, Any]] = []
    for s in pair_seeds:
        world = make_map(map_type=map_type, seed=s)
        start, goal = sample_start_goal(world, seed=s)
        pairs.append((start, goal))
    return pairs


# ------------------------------ Main eval -----------------------------------

def evaluate_method_on_pairs(method_name: str, map_type: str, pairs: List[Tuple[Any, Any]]) -> Metrics:
    """Run evaluation for one method on a list of fixed start-goal pairs."""

    # Expected to exist in project.
    from planners.registry import get_planner  # type: ignore
    from utils.maps import make_map  # type: ignore

    metrics = Metrics()

    # deterministic run seeds for repeated runs per pair
    run_seeds = get_seeds(50_000 + MAP_TYPES.index(map_type) * 10_000, STATIC_N)

    for i, (start, goal) in enumerate(pairs):
        for j, run_seed in enumerate(run_seeds):
            # recreate world deterministically for each pair seed index
            # (assumes make_map depends only on seed and map_type)
            pair_seed = 10_000 + MAP_TYPES.index(map_type) * 1_000 + i
            world = make_map(map_type=map_type, seed=pair_seed)

            planner = get_planner(method_name, world=world, seed=run_seed)
            result = planner.plan(start=start, goal=goal, max_steps=MAX_STEPS_STATIC)

            success, collision, path_len = extract_episode_outcome(result)

            metrics.total_count += 1
            if success:
                metrics.success_count += 1
                metrics.sum_path_length += float(path_len)
            if collision:
                metrics.collision_count += 1

    return metrics


def main() -> None:
    # Expected to exist in project.
    from planners.registry import list_planners  # type: ignore

    methods = list_planners()

    rows: List[Dict[str, Any]] = []

    txt_lines: List[str] = []
    txt_lines.append(
        f"STATIC EVAL | K={STATIC_K} pairs/map | N={STATIC_N} runs/pair | max_steps={MAX_STEPS_STATIC}"
    )

    for map_type in MAP_TYPES:
        pairs = generate_fixed_start_goal_pairs(map_type, STATIC_K)

        txt_lines.append(f"\n=== Map: {map_type} ===")
        for method in methods:
            m = evaluate_method_on_pairs(method, map_type, pairs)
            row = {
                "map_type": map_type,
                "method": method,
                "K": STATIC_K,
                "N": STATIC_N,
                "max_steps": MAX_STEPS_STATIC,
                "success_count": m.success_count,
                "collision_count": m.collision_count,
                "total": m.total_count,
                "success_rate": m.success_rate,
                "collision_rate": m.collision_rate,
                "path_length": m.avg_path_length,
            }
            rows.append(row)

            txt_lines.append(
                f"{method:>20s} | success {m.success_count}/{m.total_count} (rate={m.success_rate:.3f})"
                f" | collision {m.collision_count}/{m.total_count} (rate={m.collision_rate:.3f})"
                f" | avg_path_len={m.avg_path_length:.3f}"
            )

    # Save outputs to repo root
    csv_path = os.path.join(os.path.dirname(__file__), "eval_static_all.csv")
    txt_path = os.path.join(os.path.dirname(__file__), "eval_static_all.txt")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines) + "\n")


if __name__ == "__main__":
    main()
