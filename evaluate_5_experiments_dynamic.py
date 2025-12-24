"""Dynamic evaluation (5 experiments) with reproducible metrics.

Changes vs. previous version:
- Uses shared constants from experiments_config.py.
- Reports success_count, collision_count, collision_rate in addition to
  success_rate and path_length.
- Saves outputs to both CSV and TXT in the repository root.

Assumes project provides utilities for dynamic environments and planners.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from experiments_config import (
    MAP_TYPES,
    DYNAMIC_K,
    DYNAMIC_N,
    MAX_STEPS_DYNAMIC,
    get_seeds,
)


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
        return self.sum_path_length / self.success_count if self.success_count else 0.0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def extract_episode_outcome(result: Any):
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

    if isinstance(result, (tuple, list)) and len(result) > 0:
        success = bool(result[0])
        collision = False
        path_len = 0.0
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


def main() -> None:
    # Expected to exist in project.
    from planners.registry import list_planners, get_planner  # type: ignore
    from utils.dynamic_maps import make_dynamic_scenario  # type: ignore

    methods = list_planners()

    rows: List[Dict[str, Any]] = []
    txt_lines: List[str] = []
    txt_lines.append(
        f"DYNAMIC EVAL | K={DYNAMIC_K} scenarios/map | N={DYNAMIC_N} runs/scenario | max_steps={MAX_STEPS_DYNAMIC}"
    )

    for map_type in MAP_TYPES:
        txt_lines.append(f"\n=== Map: {map_type} ===")

        # deterministic scenario seeds per map
        scenario_base = 20_000 + MAP_TYPES.index(map_type) * 2_000
        scenario_seeds = get_seeds(scenario_base, DYNAMIC_K)

        for method in methods:
            m = Metrics()

            for s_idx, scenario_seed in enumerate(scenario_seeds):
                run_seeds = get_seeds(80_000 + MAP_TYPES.index(map_type) * 10_000 + s_idx * 100, DYNAMIC_N)

                for run_seed in run_seeds:
                    scenario = make_dynamic_scenario(map_type=map_type, seed=scenario_seed)
                    planner = get_planner(method, world=scenario.world, seed=run_seed)

                    # expect scenario provides start/goal and dynamic obstacles via world
                    result = planner.plan(
                        start=scenario.start,
                        goal=scenario.goal,
                        max_steps=MAX_STEPS_DYNAMIC,
                        dynamic_obstacles=getattr(scenario, "dynamic_obstacles", None),
                    )

                    success, collision, path_len = extract_episode_outcome(result)
                    m.total_count += 1
                    if success:
                        m.success_count += 1
                        m.sum_path_length += float(path_len)
                    if collision:
                        m.collision_count += 1

            row = {
                "map_type": map_type,
                "method": method,
                "K": DYNAMIC_K,
                "N": DYNAMIC_N,
                "max_steps": MAX_STEPS_DYNAMIC,
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

    csv_path = os.path.join(os.path.dirname(__file__), "eval_dynamic_5exp.csv")
    txt_path = os.path.join(os.path.dirname(__file__), "eval_dynamic_5exp.txt")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines) + "\n")


if __name__ == "__main__":
    main()
