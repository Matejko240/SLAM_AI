import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_slam_ai.common import (
    merge_balanced_indices,
    scan_motion_confidence,
    split_time_coverage_stats,
    split_train_val_indices,
    uniform_histogram_sample_indices,
    velocity_label_from_poses,
    wait_for_npz_dataset,
)


class WaitForNpzDatasetTests(unittest.TestCase):
    def test_wait_for_npz_dataset_accepts_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.npz"
            np.savez_compressed(
                dataset_path,
                X_scan=np.zeros((4, 360), dtype=np.float32),
                X_odom=np.zeros((4, 3), dtype=np.float32),
                Y=np.zeros((4, 3), dtype=np.float32),
            )
            ready, error = wait_for_npz_dataset(
                str(dataset_path),
                timeout_sec=0.5,
                required_keys=("X_scan", "X_odom", "Y"),
            )
            self.assertTrue(ready)
            self.assertIsNone(error)

    def test_wait_for_npz_dataset_rejects_missing_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset_missing_keys.npz"
            np.savez_compressed(dataset_path, X_scan=np.zeros((2, 360), dtype=np.float32))
            ready, error = wait_for_npz_dataset(
                str(dataset_path),
                timeout_sec=0.2,
                poll_interval_sec=0.05,
                required_keys=("X_scan", "X_odom", "Y"),
            )
            self.assertFalse(ready)
            self.assertIsInstance(error, KeyError)


class SplitStrategyTests(unittest.TestCase):
    def test_tail_holdout_split_is_contiguous(self) -> None:
        train_idx, val_idx, strategy = split_train_val_indices(
            10,
            0.3,
            seed=123,
            split_strategy="tail_holdout_no_shuffle",
        )
        self.assertEqual(strategy, "tail_holdout_no_shuffle")
        self.assertTrue(np.array_equal(train_idx, np.arange(0, 7, dtype=np.int64)))
        self.assertTrue(np.array_equal(val_idx, np.arange(7, 10, dtype=np.int64)))
        stats = split_time_coverage_stats(10, train_idx, val_idx, split_strategy=strategy)
        self.assertAlmostEqual(stats["train_time_window_start_ratio"], 0.0)
        self.assertAlmostEqual(stats["val_time_window_end_ratio"], 1.0)

    def test_random_split_is_reproducible_for_seed(self) -> None:
        tr_1, va_1, strategy_1 = split_train_val_indices(30, 0.2, seed=7, split_strategy="random_shuffle")
        tr_2, va_2, strategy_2 = split_train_val_indices(30, 0.2, seed=7, split_strategy="random")
        self.assertEqual(strategy_1, "random_shuffle")
        self.assertEqual(strategy_2, "random_shuffle")
        self.assertTrue(np.array_equal(tr_1, tr_2))
        self.assertTrue(np.array_equal(va_1, va_2))


class MotionSignalTests(unittest.TestCase):
    def test_velocity_label_from_poses_uses_local_forward_motion(self) -> None:
        v, w = velocity_label_from_poses(
            (0.0, 0.0, math.pi / 2.0),
            (0.0, 1.0, math.pi),
            2.0,
            frame="local",
        )
        self.assertAlmostEqual(v, 0.5, places=5)
        self.assertAlmostEqual(w, math.pi / 4.0, places=5)

    def test_scan_motion_confidence_grows_with_delta_scan_energy(self) -> None:
        still = np.ones((360,), dtype=np.float32)
        moved = np.ones((360,), dtype=np.float32) * 1.2
        self.assertAlmostEqual(scan_motion_confidence(still, still), 0.0, places=6)
        self.assertGreater(scan_motion_confidence(still, moved), 0.5)


class HistogramBalancingTests(unittest.TestCase):
    def test_histogram_balancing_uses_configured_range_and_upsample(self) -> None:
        values = np.concatenate(
            [
                np.full((50,), 0.05, dtype=np.float32),
                np.full((40,), 0.15, dtype=np.float32),
                np.full((5,), 0.25, dtype=np.float32),
                np.full((3,), 1.2, dtype=np.float32),  # poza zakresem
            ]
        )
        selected_idx, stats = uniform_histogram_sample_indices(
            values,
            bins=3,
            seed=123,
            use_abs=False,
            hist_min=0.0,
            hist_max=0.3,
            target_quantile=0.0,
            target_min_per_bin=20,
            upsample=True,
        )

        self.assertEqual(int(stats["n_out_of_range"]), 3)
        self.assertEqual(int(stats["target_per_bin"]), 20)
        self.assertEqual(selected_idx.size, 60)
        self.assertTrue(np.array_equal(np.asarray(stats["counts_per_bin"]), np.asarray([50, 40, 5])))
        self.assertTrue(
            np.array_equal(
                np.asarray(stats["selected_counts_per_bin"]),
                np.asarray([20, 20, 20]),
            )
        )

    def test_histogram_balancing_respects_use_abs(self) -> None:
        values = np.asarray([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float32)
        selected_idx, stats = uniform_histogram_sample_indices(
            values,
            bins=2,
            seed=7,
            use_abs=True,
            hist_min=0.0,
            hist_max=2.0,
            target_quantile=0.0,
            target_min_per_bin=2,
            upsample=False,
        )
        self.assertEqual(selected_idx.size, 4)
        self.assertEqual(int(stats["bins_non_empty"]), 2)
        self.assertTrue(np.array_equal(np.asarray(stats["selected_counts_per_bin"]), np.asarray([2, 2])))

    def test_merge_balanced_indices_strategies(self) -> None:
        a = np.asarray([1, 2, 2, 5], dtype=np.int64)
        b = np.asarray([2, 3, 5, 5], dtype=np.int64)
        self.assertTrue(
            np.array_equal(
                merge_balanced_indices(a, b, strategy="union_unique"),
                np.asarray([1, 2, 3, 5], dtype=np.int64),
            )
        )
        self.assertTrue(
            np.array_equal(
                merge_balanced_indices(a, b, strategy="component_concat"),
                np.asarray([1, 2, 2, 5, 2, 3, 5, 5], dtype=np.int64),
            )
        )
        self.assertTrue(
            np.array_equal(
                merge_balanced_indices(a, b, strategy="intersection"),
                np.asarray([2, 5], dtype=np.int64),
            )
        )


if __name__ == "__main__":
    unittest.main()
