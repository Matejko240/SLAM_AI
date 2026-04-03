import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_slam_eval.eval_node import baseline_rmse_on_timeline, load_pgm


class LoadPgmTests(unittest.TestCase):
    def test_load_pgm_p2_accepts_exact_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pgm_path = Path(tmpdir) / "map_ok.pgm"
            pgm_path.write_text("P2\n2 2\n255\n1 2 3 4\n", encoding="ascii")
            arr = load_pgm(str(pgm_path))
            self.assertEqual(arr.shape, (2, 2))
            self.assertTrue(np.array_equal(arr, np.array([[1, 2], [3, 4]], dtype=np.uint8)))

    def test_load_pgm_p2_rejects_extra_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pgm_path = Path(tmpdir) / "map_bad.pgm"
            pgm_path.write_text("P2\n2 2\n255\n1 2 3 4 99\n", encoding="ascii")
            with self.assertRaises(ValueError):
                _ = load_pgm(str(pgm_path))


class CommonTimelineRmseTests(unittest.TestCase):
    def test_baseline_rmse_on_timeline_matches_common_samples(self) -> None:
        baseline_ts = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        baseline_err_xy = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float32,
        )
        baseline_err_theta = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        ai_ts = np.array([0.1, 0.3], dtype=np.float32)

        rmse_xy, rmse_th, n_common = baseline_rmse_on_timeline(
            baseline_ts,
            baseline_err_xy,
            baseline_err_theta,
            ai_ts,
        )
        self.assertEqual(n_common, 2)
        self.assertAlmostEqual(rmse_xy, float(np.sqrt((2.0**2 + 4.0**2) / 2.0)))
        self.assertAlmostEqual(rmse_th, float(np.sqrt((0.2**2 + 0.4**2) / 2.0)))

    def test_baseline_rmse_on_timeline_returns_none_without_overlap(self) -> None:
        rmse_xy, rmse_th, n_common = baseline_rmse_on_timeline(
            np.array([0.0, 0.1], dtype=np.float32),
            np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
        )
        self.assertIsNone(rmse_xy)
        self.assertIsNone(rmse_th)
        self.assertEqual(n_common, 0)


if __name__ == "__main__":
    unittest.main()
