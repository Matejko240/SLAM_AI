import importlib.util
import unittest
from pathlib import Path

_test_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "collision_visual_validation",
    _test_dir / "collision_visual_validation.py",
)
_cvv = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cvv)
validate_models_dir = _cvv.validate_models_dir
validate_sdf_links = _cvv.validate_sdf_links


class GazeboSmokeTests(unittest.TestCase):
    def test_smoke_ai_slam_gazebo_package(self) -> None:
        self.assertTrue(True)

    def test_collision_visual_same_as_visual_in_models_and_worlds(self) -> None:
        """Każda collision ma parę visual z identyczną geometrią i pose (office + hospital + modele)."""
        pkg = Path(__file__).resolve().parents[1]
        models = pkg / "models"
        worlds = pkg / "worlds"
        errs = validate_models_dir(models)
        for w in ("world_office.sdf", "world_hospital.sdf"):
            p = worlds / w
            if p.is_file():
                errs.extend(validate_sdf_links(p))
        self.assertEqual(
            errs,
            [],
            "collision/visual mismatch:\n" + "\n".join(errs[:50]) + (f"\n… ({len(errs)} total)" if len(errs) > 50 else ""),
        )


if __name__ == "__main__":
    unittest.main()
