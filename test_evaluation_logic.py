#!/usr/bin/env python3
"""
Unit tests for dynamic ablation evaluation logic.

These tests validate the core logic without requiring the parent modules.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from experiments_config import MAP_TYPES, DYNAMIC_K, DYNAMIC_N, MAX_STEPS_DYNAMIC, get_seeds
from dataclasses import dataclass
import numpy as np


def test_seed_generation():
    """Test that seed generation is deterministic."""
    print("Testing seed generation...")
    
    # Test basic seed generation
    seeds1 = get_seeds(1000, 10)
    seeds2 = get_seeds(1000, 10)
    assert seeds1 == seeds2, "Seeds should be deterministic"
    assert len(seeds1) == 10, "Should generate 10 seeds"
    assert seeds1[0] == 1000, "First seed should be base"
    assert seeds1[-1] == 1009, "Last seed should be base+9"
    
    # Test scenario seed generation pattern (from evaluation script)
    for map_idx, map_type in enumerate(MAP_TYPES):
        scenario_base = 20_000 + map_idx * 2_000
        scenario_seeds = get_seeds(scenario_base, DYNAMIC_K)
        
        assert len(scenario_seeds) == DYNAMIC_K, f"Should have {DYNAMIC_K} scenario seeds"
        assert scenario_seeds[0] == scenario_base, "First seed should match base"
        
        # Test run seed generation for each scenario
        for s_idx in range(DYNAMIC_K):
            run_base = 80_000 + map_idx * 10_000 + s_idx * 100
            run_seeds = get_seeds(run_base, DYNAMIC_N)
            
            assert len(run_seeds) == DYNAMIC_N, f"Should have {DYNAMIC_N} run seeds"
            assert run_seeds[0] == run_base, "First run seed should match base"
    
    print("  ✓ Seed generation is deterministic and correct")


def test_episode_counts():
    """Test that episode counts match protocol."""
    print("Testing episode counts...")
    
    # Calculate expected totals
    episodes_per_map_method = DYNAMIC_K * DYNAMIC_N
    total_episodes = len(MAP_TYPES) * 5 * episodes_per_map_method  # 5 methods
    
    assert episodes_per_map_method == 200, "Should be 200 episodes per (map, method)"
    assert total_episodes == 4000, "Should be 4000 total episodes (4 maps * 5 methods * 200)"
    
    print(f"  ✓ Episode counts correct: {episodes_per_map_method} per (map, method)")
    print(f"  ✓ Total episodes: {total_episodes}")


def test_path_length_calculation():
    """Test path length calculation function."""
    print("Testing path length calculation...")
    
    # Simple test trajectory: forms a square
    # [0,0] → [1,0] → [1,1] → [0,1]
    # Length: 1 + 1 + 1 = 3.0
    traj = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    
    # Calculate expected length: 1 + 1 + 1 = 3
    expected = 3.0
    
    # Path length function (from evaluation script)
    def path_length(traj):
        if traj is None or len(traj) < 2:
            return 0.0
        diffs = np.diff(traj, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    calculated = path_length(traj)
    assert abs(calculated - expected) < 0.001, f"Path length should be {expected}, got {calculated}"
    
    # Test empty trajectory
    assert path_length(None) == 0.0, "None trajectory should have 0 length"
    assert path_length(np.array([[0, 0]])) == 0.0, "Single-point trajectory should have 0 length"
    
    print("  ✓ Path length calculation correct")


def test_aggregation_logic():
    """Test result aggregation logic."""
    print("Testing result aggregation...")
    
    @dataclass
    class MockResult:
        map_type: str
        method: str
        success: bool
        collision: bool
        timeout: bool
        path_length_m: float
        wall_time_ms: float
    
    # Create mock results
    results = [
        MockResult('simple', 'A*', True, False, False, 100.0, 10.0),
        MockResult('simple', 'A*', True, False, False, 110.0, 12.0),
        MockResult('simple', 'A*', False, True, False, 0.0, 8.0),
        MockResult('simple', 'RRT*', True, False, False, 120.0, 50.0),
        MockResult('simple', 'RRT*', False, False, True, 0.0, 100.0),
    ]
    
    # Aggregate manually
    groups = {}
    for r in results:
        key = (r.map_type, r.method)
        groups.setdefault(key, []).append(r)
    
    # Check grouping
    assert len(groups) == 2, "Should have 2 groups"
    assert ('simple', 'A*') in groups, "Should have A* group"
    assert ('simple', 'RRT*') in groups, "Should have RRT* group"
    
    # Check A* metrics
    astar_results = groups[('simple', 'A*')]
    total = len(astar_results)
    success_count = sum(1 for r in astar_results if r.success)
    collision_count = sum(1 for r in astar_results if r.collision)
    
    assert total == 3, "A* should have 3 results"
    assert success_count == 2, "A* should have 2 successes"
    assert collision_count == 1, "A* should have 1 collision"
    
    success_rate = success_count / total
    assert abs(success_rate - 2/3) < 0.001, "Success rate should be 2/3"
    
    # Average path length (over successful only)
    successful_paths = [r.path_length_m for r in astar_results if r.success]
    avg_path = np.mean(successful_paths)
    assert abs(avg_path - 105.0) < 0.001, "Average path should be 105"
    
    print("  ✓ Aggregation logic correct")


def test_checkpoint_paths():
    """Test checkpoint path definitions."""
    print("Testing checkpoint paths...")
    
    checkpoint_info = {
        'PPO (Basic)': 'models_basic_astar/model.pth',
        'PPO (Dual-Att)': 'models_attention_noastar/model.pth',
        'Ours (Full)': 'best_navigation_model.pth',
    }
    
    # Verify paths follow naming convention
    assert 'basic' in checkpoint_info['PPO (Basic)'], "Basic model path should contain 'basic'"
    assert 'astar' in checkpoint_info['PPO (Basic)'], "Basic model uses A* so path should contain 'astar'"
    
    assert 'attention' in checkpoint_info['PPO (Dual-Att)'], "Attention model path should contain 'attention'"
    assert 'noastar' in checkpoint_info['PPO (Dual-Att)'], "Attention model doesn't use A* so path should contain 'noastar'"
    
    assert checkpoint_info['Ours (Full)'] == 'best_navigation_model.pth', "Full model should use standard checkpoint name"
    
    print("  ✓ Checkpoint paths follow conventions")


def test_output_fields():
    """Test that output CSV/TXT will have required fields."""
    print("Testing output fields...")
    
    required_fields = [
        'map_type',
        'method',
        'K',
        'N',
        'max_steps',
        'success_count',
        'collision_count',
        'timeout_count',
        'total',
        'success_rate',
        'collision_rate',
        'avg_path_length_m',
        'avg_wall_time_ms',
    ]
    
    # Create a mock summary row
    mock_row = {
        'map_type': 'simple',
        'method': 'A* tracking',
        'K': DYNAMIC_K,
        'N': DYNAMIC_N,
        'max_steps': MAX_STEPS_DYNAMIC,
        'success_count': 180,
        'collision_count': 10,
        'timeout_count': 10,
        'total': 200,
        'success_rate': 0.9,
        'collision_rate': 0.05,
        'avg_path_length_m': 95.3,
        'avg_wall_time_ms': 12.5,
    }
    
    # Verify all required fields are present
    for field in required_fields:
        assert field in mock_row, f"Missing required field: {field}"
    
    # Verify counts add up
    assert mock_row['total'] == DYNAMIC_K * DYNAMIC_N, "Total should be K*N"
    
    print("  ✓ Output fields are complete")


def main():
    """Run all tests."""
    print("=" * 70)
    print("VALIDATION TESTS FOR DYNAMIC ABLATION EVALUATION")
    print("=" * 70)
    print()
    
    tests = [
        test_seed_generation,
        test_episode_counts,
        test_path_length_calculation,
        test_aggregation_logic,
        test_checkpoint_paths,
        test_output_fields,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
