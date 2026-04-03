import math
import unittest
from pathlib import Path

import numpy as np
import yaml

from ai_slam_bringup.occupancy_grid_plan import (
    astar,
    densify_polyline,
    inflate_obstacles,
    load_reference_map_layers,
    plan_polyline_through_anchors,
    remove_grid_loops,
)
from ai_slam_bringup.planned_path_driver import (
    _lookahead,
    _resolve_turn_direction_sign,
    _update_turn_in_place_mode,
)
from ai_slam_bringup.dataset_motion_watchdog import (
    DatasetMotionWatchdog,
    _is_low_progress_stall,
    _recent_motion_metrics,
)


class BringupSmokeTests(unittest.TestCase):
    def test_smoke_ai_slam_bringup_package(self) -> None:
        self.assertTrue(True)

    def test_smoke_dataset_motion_watchdog_module(self) -> None:
        self.assertTrue(callable(DatasetMotionWatchdog))

    def test_motion_watchdog_recent_metrics_for_jitter(self) -> None:
        # 40s mikro-ruchu wokół punktu: duży czas okna, niski net progress.
        samples = []
        for i in range(401):
            t = float(i) * 0.1
            x = 1.0 + 0.01 * math.sin(float(i) * 0.3)
            y = -2.0 + 0.01 * math.cos(float(i) * 0.25)
            samples.append((t, x, y))
        duration, net, span, count = _recent_motion_metrics(samples, now_sec=40.0, window_sec=35.0)
        self.assertGreaterEqual(duration, 34.0)
        self.assertGreaterEqual(count, 300)
        self.assertLess(net, 0.06)
        self.assertLess(span, 0.06)

    def test_motion_watchdog_low_progress_stall_detects_jitter(self) -> None:
        samples = []
        for i in range(401):
            t = float(i) * 0.1
            x = 0.5 + 0.01 * math.sin(float(i) * 0.2)
            y = 0.2 + 0.01 * math.cos(float(i) * 0.4)
            samples.append((t, x, y))
        stalled, duration, net, span, _ = _is_low_progress_stall(
            samples,
            now_sec=40.0,
            window_sec=35.0,
            min_progress_m=0.12,
            span_ratio=1.8,
        )
        self.assertTrue(stalled, msg=f"duration={duration:.2f}, net={net:.4f}, span={span:.4f}")

    def test_motion_watchdog_low_progress_stall_ignores_real_progress(self) -> None:
        # Realny ruch w korytarzu: watchdog nie może tego oznaczyć jako stall.
        samples = []
        for i in range(401):
            t = float(i) * 0.1
            x = 0.04 * t  # 1.6m w 40s
            y = 0.02 * math.sin(0.4 * t)
            samples.append((t, x, y))
        stalled, _, net, span, _ = _is_low_progress_stall(
            samples,
            now_sec=40.0,
            window_sec=35.0,
            min_progress_m=0.12,
            span_ratio=1.8,
        )
        self.assertFalse(stalled, msg=f"net={net:.4f}, span={span:.4f}")

    def test_astar_corridor(self) -> None:
        h, w = 7, 7
        walk = np.ones((h, w), dtype=bool)
        walk[:, 3] = False
        walk[3, :] = True
        p = astar(walk, (3, 0), (3, 6))
        self.assertIsNotNone(p)
        self.assertEqual(p[0], (3, 0))
        self.assertEqual(p[-1], (3, 6))

    def test_densify_polyline(self) -> None:
        d = densify_polyline([(0.0, 0.0), (1.0, 0.0)], 0.3)
        self.assertGreaterEqual(len(d), 3)

    def test_inflate_obstacles(self) -> None:
        occ = np.zeros((5, 5), dtype=bool)
        occ[2, 2] = True
        out = inflate_obstacles(occ, 1)
        self.assertTrue(out[1, 2])

    def test_remove_grid_loops(self) -> None:
        raw = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 0), (2, 0), (3, 0)]
        cleaned = remove_grid_loops(raw)
        self.assertEqual(len(cleaned), len(set(cleaned)))
        self.assertEqual(cleaned[0], (0, 0))
        self.assertEqual(cleaned[-1], (3, 0))

    def test_lookahead_limits_nearest_search_to_local_horizon(self) -> None:
        # Trasa wraca bardzo blisko startu; globalne nearest-search powodowało skok indeksu.
        path = [
            (0.00, 0.00),
            (1.00, 0.00),
            (2.00, 0.00),
            (2.00, 1.00),
            (1.00, 1.00),
            (0.05, 0.05),  # odległa część trasy, ale blisko geometrycznie do startu
            (0.00, 1.00),
        ]
        _, _, idx = _lookahead(
            path,
            px=0.02,
            py=0.02,
            start_idx=0,
            lookahead_m=0.30,
            nearest_backtrack_points=4,
            nearest_horizon_m=1.0,
        )
        self.assertLessEqual(idx, 1, msg=f"unexpected far jump idx={idx}")

    def test_lookahead_keeps_progress_when_already_advanced(self) -> None:
        path = [
            (0.00, 0.00),
            (1.00, 0.00),
            (2.00, 0.00),
            (2.00, 1.00),
            (1.00, 1.00),
            (0.05, 0.05),
            (0.00, 1.00),
        ]
        _, _, idx = _lookahead(
            path,
            px=1.95,
            py=0.95,
            start_idx=3,
            lookahead_m=0.30,
            nearest_backtrack_points=4,
            nearest_horizon_m=1.0,
        )
        self.assertGreaterEqual(idx, 3, msg=f"unexpected regression idx={idx}")

    def test_turn_direction_sign_stays_stable_near_pi_wrap(self) -> None:
        sign = 0.0
        guard = math.radians(18.0)
        # Symuluje jitter wokół +/-pi, który wcześniej potrafił zmieniać znak obrotu co tick.
        for err in [3.13, -3.13, 3.12, -3.12, 3.10]:
            sign = _resolve_turn_direction_sign(err, abs(err), sign, guard)
        self.assertEqual(sign, 1.0)

    def test_turn_direction_sign_uses_preference_on_ambiguous_first_tick(self) -> None:
        guard = math.radians(18.0)
        # Pierwszy tick: err ~ -pi (niejednoznaczny znak po wrap), preferencja ma wygrać.
        sign_left = _resolve_turn_direction_sign(-3.13, 3.13, 0.0, guard, preferred_sign=1.0)
        sign_right = _resolve_turn_direction_sign(-3.13, 3.13, 0.0, guard, preferred_sign=-1.0)
        self.assertEqual(sign_left, 1.0)
        self.assertEqual(sign_right, -1.0)

    def test_turn_in_place_mode_uses_hysteresis(self) -> None:
        stop = math.radians(55.0)
        resume = math.radians(35.0)
        mode = False
        mode = _update_turn_in_place_mode(mode, math.radians(70.0), stop, resume)
        self.assertTrue(mode)
        mode = _update_turn_in_place_mode(mode, math.radians(45.0), stop, resume)
        self.assertTrue(mode, msg="mode should stay active until resume threshold")
        mode = _update_turn_in_place_mode(mode, math.radians(20.0), stop, resume)
        self.assertFalse(mode)

    def test_long_trajectory_specs_plan_on_reference_maps(self) -> None:
        """Pliki z scripts/generate_long_trajectories.py: kotwice + A* na mapie ref. bez błędu."""
        pkg = Path(__file__).resolve().parents[1]
        maps = pkg.parent / "ai_slam_eval" / "maps"
        planned = pkg / "config" / "planned_paths"
        specs = [
            ("office_trajectory_acyclic.yaml", "reference_map_office.yaml"),
            ("office_trajectory_cyclic_2lap.yaml", "reference_map_office.yaml"),
            ("hospital_trajectory_acyclic.yaml", "reference_map_hospital.yaml"),
            ("hospital_trajectory_cyclic_2lap.yaml", "reference_map_hospital.yaml"),
        ]
        for spec_name, map_name in specs:
            sp = planned / spec_name
            mp = maps / map_name
            if not sp.is_file() or not mp.is_file():
                continue
            with open(sp, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f) or {}
            anchors_raw = spec.get("anchors") or []
            anchors = [(float(a["x"]), float(a["y"])) for a in anchors_raw if isinstance(a, dict)]
            self.assertGreaterEqual(len(anchors), 2, msg=spec_name)
            pgm, blocked, meta = load_reference_map_layers(str(mp))
            res = float(meta["resolution"])
            inflate_m = float(spec.get("inflate_robot_m", 0.35))
            inflate_cells = max(1, int(math.ceil(inflate_m / res)))
            poly = plan_polyline_through_anchors(
                anchors,
                blocked,
                meta,
                flip_y=bool(spec.get("map_flip_y", True)),
                inflate_cells=inflate_cells,
            )
            lp = sum(
                math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
                for i in range(len(poly) - 1)
            )
            self.assertGreater(lp, 15.0, msg=f"{spec_name}: A* poly too short")


if __name__ == "__main__":
    unittest.main()
