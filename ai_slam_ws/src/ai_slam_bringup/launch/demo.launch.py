"""
AI SLAM Demo Launch File - z centralną konfiguracją

Izolacja torów (badania):
  Baseline / AI / Robak / Rywak: osobne topic skanu, osobne instancje slam_toolbox i map.
  Inferencja metod NIE używa kotwicy scanmatch (usunięte) — tylko model + odom/GT wg węzła.

Użycie:
  # Domyślna konfiguracja:
  ros2 launch ai_slam_bringup demo.launch.py
  
  # Własna konfiguracja:
  ros2 launch ai_slam_bringup demo.launch.py config:=experiment_config.yaml
  ros2 launch ai_slam_bringup demo.launch.py config:=fast_test.yaml
  
  # Override pojedynczych parametrów:
  ros2 launch ai_slam_bringup demo.launch.py mode:=baseline duration_sec:=60
  ros2 launch ai_slam_bringup demo.launch.py config:=fast_test.yaml seed:=999

Czas datasetu / ewaluacji (wspólny dla torów): pipeline.dataset_collection_sec, pipeline.evaluation_sec
(w YAML); bez pipeline — timing.dataset_duration / timing.eval_duration. Timeouty oczekiwania na dataset
i model skaluje lifecycle (max( wartość_z_yaml, szacunek )) — przy 0 w YAML zostaje tylko szacunek.

Sterowanie: driver.use_planned_path true → planned_path_driver (kotwice + opcjonalnie A* na mapie ref.),
false → auto_driver (reaktywny / skryptowany).
"""
import os
import yaml
import math
import re
import copy
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from launch.event_handlers import OnProcessExit
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, TimerAction, SetEnvironmentVariable, 
    IncludeLaunchDescription, EmitEvent, LogInfo, RegisterEventHandler, ExecuteProcess,
    OpaqueFunction, GroupAction
)
from launch.conditions import IfCondition, UnlessCondition
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from ament_index_python.packages import get_package_share_directory


def generate_experiment_id() -> str:
    """Generuje unikalny identyfikator eksperymentu."""
    return "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_config_reference(ref: str, bringup_share: str, current_dir: str) -> str:
    if os.path.isabs(ref):
        return os.path.abspath(ref)
    local_candidate = os.path.abspath(os.path.join(current_dir, ref))
    if os.path.exists(local_candidate):
        return local_candidate
    return os.path.abspath(os.path.join(bringup_share, "config", ref))


def _load_config_recursive(config_path: str, bringup_share: str, stack: tuple[str, ...]) -> dict:
    abs_path = os.path.abspath(config_path)
    if abs_path in stack:
        cycle_chain = " -> ".join([*stack, abs_path])
        raise RuntimeError(f"Config extends cycle detected: {cycle_chain}")
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping/object: {abs_path}")

    extends = data.pop("extends", None)
    if not extends:
        return data

    if isinstance(extends, str):
        parents = [extends]
    elif isinstance(extends, list):
        parents = [str(item) for item in extends if str(item).strip()]
    else:
        raise ValueError(f"Invalid 'extends' type in {abs_path}: expected string or list")

    merged_parent = {}
    for parent_ref in parents:
        parent_path = _resolve_config_reference(parent_ref, bringup_share, os.path.dirname(abs_path))
        parent_cfg = _load_config_recursive(parent_path, bringup_share, (*stack, abs_path))
        merged_parent = _deep_merge_dicts(merged_parent, parent_cfg)

    return _deep_merge_dicts(merged_parent, data)


def load_config(config_file: str) -> dict:
    """Wczytuje konfigurację z pliku YAML (obsługuje optional 'extends')."""
    bringup_share = get_package_share_directory("ai_slam_bringup")

    if not config_file:
        return {}

    # Jeśli podano tylko nazwę pliku, szukaj w config/
    if not os.path.isabs(config_file):
        config_file = os.path.join(bringup_share, "config", config_file)

    if not os.path.exists(config_file):
        return {}

    return _load_config_recursive(config_file, bringup_share, ())


def resolve_config_path(config_file: str) -> str:
    bringup_share = get_package_share_directory("ai_slam_bringup")
    if not config_file:
        return ""
    if not os.path.isabs(config_file):
        config_file = os.path.join(bringup_share, "config", config_file)
    return os.path.abspath(config_file)


def get_config_value(config: dict, *keys, default=None):
    """Bezpiecznie pobiera wartość z zagnieżdżonego słownika."""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def driver_cfg_float(config: dict, new_key: str, old_key: str, default: float) -> float:
    """Preferuje nowe klucze driver.* (np. no_cycle_*), z fallbackiem na fixed_*."""
    v = get_config_value(config, "driver", new_key, default=None)
    if v is not None and str(v).strip() != "":
        return float(v)
    v2 = get_config_value(config, "driver", old_key, default=None)
    if v2 is not None and str(v2).strip() != "":
        return float(v2)
    return float(default)


def driver_cfg_int(config: dict, new_key: str, old_key: str, default: int) -> int:
    v = get_config_value(config, "driver", new_key, default=None)
    if v is not None and str(v).strip() != "":
        return int(v)
    v2 = get_config_value(config, "driver", old_key, default=None)
    if v2 is not None and str(v2).strip() != "":
        return int(v2)
    return int(default)


def parse_bool(value, default=False):
    """Konwertuje bool/str/int na bool w sposób odporny na 'false' jako string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "t", "yes", "y", "on"):
            return True
        if v in ("0", "false", "f", "no", "n", "off", ""):
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def normalize_dataset_trajectory_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"no_cycle", "nocycle", "acyclic", "without_cycles"}:
        return "no_cycle"
    if mode in {"cycle", "cyclic", "with_cycles"}:
        return "cycle"
    return "any"


def normalize_balance_merge_strategy(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"component_concat", "concat", "components", "sum"}:
        return "component_concat"
    if mode in {"intersection", "intersect"}:
        return "intersection"
    return "union_unique"


def normalize_driver_trajectory_mode(value: str) -> str:
    """Tryb sterownika: auto | no_cycle | cycle (aliasy fixed_* nadal działają)."""
    mode = str(value).strip().lower()
    if mode in {"", "auto", "auto_nav", "autodrive", "random", "any"}:
        return "auto"
    if mode in {"no_cycle", "nocycle", "acyclic", "without_cycles", "fixed_no_cycle"}:
        return "no_cycle"
    if mode in {"cycle", "cyclic", "with_cycles", "fixed_cycle"}:
        return "cycle"
    return "auto"


def merge_params(*param_dicts):
    """Łączy słowniki parametrów, ignorując wartości niebędące dict."""
    merged = {}
    for params in param_dicts:
        if isinstance(params, dict):
            merged.update(params)
    return merged


def coerce_slam_param_types(params: dict) -> dict:
    """Wymusza zgodne typy parametrów slam_toolbox dla wartości z YAML configu."""
    if not isinstance(params, dict):
        return {}

    float_keys = {
        "resolution",
        "max_laser_range",
        "minimum_time_interval",
        "minimum_travel_distance",
        "minimum_travel_heading",
        "transform_timeout",
        "tf_buffer_duration",
        "map_update_interval",
    }

    normalized = {}
    for key, value in params.items():
        if key in float_keys and isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def extract_world_name(world_path: str) -> str:
    """Próbuje odczytać nazwę świata z pliku SDF, fallback: nazwa pliku bez rozszerzenia."""
    fallback = os.path.splitext(os.path.basename(world_path))[0] or "default"
    try:
        with open(world_path, "r", encoding="utf-8") as f:
            txt = f.read(20000)
        m = re.search(r"<world\s+name\s*=\s*[\"']([^\"']+)[\"']", txt)
        if m:
            return str(m.group(1))
    except Exception:
        pass
    return fallback


DEFAULT_WORLD_SPAWN_POSES = {
    "world_house.sdf": {"x": 5.0, "y": 0.0, "z": 0.10, "yaw": 0.0},
    "world_house": {"x": 5.0, "y": 0.0, "z": 0.10, "yaw": 0.0},
    "world_office.sdf": {"x": 0.03, "y": 2.27, "z": 0.10, "yaw": 0.0},
    "world_office": {"x": 0.03, "y": 2.27, "z": 0.10, "yaw": 0.0},
    "world_hospital.sdf": {"x": 0.00, "y": -25.00, "z": 0.10, "yaw": 0.0},
    "world_hospital": {"x": 0.00, "y": -25.00, "z": 0.10, "yaw": 0.0},
}


def world_aliases(requested_world: str, world_path: str, world_name: str) -> list[str]:
    aliases = []
    candidates = [
        requested_world,
        os.path.basename(requested_world) if requested_world else "",
        os.path.splitext(os.path.basename(requested_world))[0] if requested_world else "",
        world_path,
        os.path.basename(world_path) if world_path else "",
        os.path.splitext(os.path.basename(world_path))[0] if world_path else "",
        world_name,
    ]
    for candidate in candidates:
        candidate = str(candidate).strip()
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    return aliases


def _coerce_spawn_value(value, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return fallback
    return fallback


def resolve_spawn_pose(sim_cfg: dict, requested_world: str, world_path: str, world_name: str) -> dict:
    aliases = world_aliases(requested_world, world_path, world_name)
    spawn_poses_cfg = sim_cfg.get("spawn_poses", {}) if isinstance(sim_cfg, dict) else {}
    spawn_pose_cfg = sim_cfg.get("spawn_pose", {}) if isinstance(sim_cfg, dict) else {}

    base_pose = {"x": 0.0, "y": 0.0, "z": 0.10, "yaw": 0.0}
    for alias in aliases:
        if alias in DEFAULT_WORLD_SPAWN_POSES:
            base_pose = dict(DEFAULT_WORLD_SPAWN_POSES[alias])
            break

    if isinstance(spawn_poses_cfg, dict):
        for alias in aliases:
            candidate = spawn_poses_cfg.get(alias)
            if isinstance(candidate, dict):
                base_pose.update(candidate)
                break

    if isinstance(spawn_pose_cfg, dict):
        base_pose.update(spawn_pose_cfg)

    return {
        "x": _coerce_spawn_value(base_pose.get("x"), 0.0),
        "y": _coerce_spawn_value(base_pose.get("y"), 0.0),
        "z": _coerce_spawn_value(base_pose.get("z"), 0.10),
        "yaw": _coerce_spawn_value(base_pose.get("yaw"), 0.0),
    }


def build_world_with_embedded_robot(base_world_path: str, model_sdf_path: str, robot_name: str, spawn_pose: dict) -> str:
    """Tworzy tymczasowy plik świata z osadzonym modelem robota już na starcie Gazebo."""
    world_tree = ET.parse(base_world_path)
    world_root = world_tree.getroot()
    world_elem = world_root.find("world")
    if world_elem is None:
        raise RuntimeError(f"World element not found in SDF: {base_world_path}")

    model_tree = ET.parse(model_sdf_path)
    model_root = model_tree.getroot()
    model_elem = model_root.find("model")
    if model_elem is None:
        raise RuntimeError(f"Model element not found in SDF: {model_sdf_path}")

    for child in list(world_elem):
        child_name = child.get("name", "").strip()
        if child.tag == "model" and child_name == robot_name:
            world_elem.remove(child)

    embedded_model = copy.deepcopy(model_elem)
    embedded_model.set("name", robot_name)

    pose_text = (
        f"{spawn_pose['x']:.6f} "
        f"{spawn_pose['y']:.6f} "
        f"{spawn_pose['z']:.6f} "
        f"0 0 {spawn_pose['yaw']:.6f}"
    )
    pose_elem = embedded_model.find("pose")
    if pose_elem is None:
        pose_elem = ET.Element("pose")
        embedded_model.insert(0, pose_elem)
    pose_elem.text = pose_text

    world_elem.append(embedded_model)

    if hasattr(ET, "indent"):
        ET.indent(world_tree, space="  ")

    tmp = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f"embedded_{os.path.splitext(os.path.basename(base_world_path))[0]}_",
        suffix=".sdf",
        delete=False,
    )
    tmp_path = tmp.name
    tmp.close()
    world_tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    return tmp_path


def source_package_dir(repo_root: str, package_name: str) -> str:
    candidate = os.path.join(repo_root, "ai_slam_ws", "src", package_name)
    return candidate if os.path.isdir(candidate) else ""


def prefer_source_path(source_root: str, install_root: str, *parts: str) -> str:
    rel_parts = [str(part).lstrip("/\\") for part in parts if str(part)]
    if source_root:
        source_candidate = os.path.join(source_root, *rel_parts)
        if os.path.exists(source_candidate):
            return source_candidate
    return os.path.join(install_root, *rel_parts)


def resolve_world_path(selected_world: str, gazebo_source_root: str, gazebo_install_root: str) -> str:
    if os.path.isabs(selected_world):
        return selected_world
    normalized = str(selected_world).strip()
    if normalized.startswith("worlds/") or normalized.startswith("worlds\\"):
        rel_parts = normalized.split("/") if "/" in normalized else normalized.split("\\")
        return prefer_source_path(gazebo_source_root, gazebo_install_root, *rel_parts)
    return prefer_source_path(gazebo_source_root, gazebo_install_root, "worlds", normalized)


def launch_setup(context, *args, **kwargs):
    """Funkcja setup wywoływana w runtime z dostępem do kontekstu."""
    
    # Pobierz wartości argumentów launch
    config_file = LaunchConfiguration("config").perform(context)
    resolved_config_path = resolve_config_path(config_file)
    
    # Wczytaj konfigurację z pliku
    cfg = load_config(config_file) if config_file else {}
    
    # Funkcja pomocnicza do pobierania parametrów (launch arg > config file > default)
    def get_param(launch_arg: str, config_keys: list, default):
        """Pobiera parametr: najpierw z launch arg, potem z config, potem default."""
        launch_val = LaunchConfiguration(launch_arg).perform(context)
        # Jeśli launch arg nie został nadpisany (ma wartość "__USE_CONFIG__"), użyj config
        if launch_val == "__USE_CONFIG__":
            return get_config_value(cfg, *config_keys, default=default)
        return launch_val
    
    # === PARAMETRY EKSPERYMENTU ===
    mode = str(get_param("mode", ["experiment", "mode"], "ai"))
    seed = int(get_param("seed", ["experiment", "seed"], 123))
    gui = str(get_param("gui", ["experiment", "gui"], "false")).lower()
    phase = str(get_param("phase", ["experiment", "phase"], "full")).lower()
    if phase not in ("full", "train", "test", "dataset"):
        phase = "full"
    world_sdf_arg = str(get_param("world_sdf", ["simulation", "world_sdf"], "__AUTO__"))
    reference_map_yaml_arg = str(LaunchConfiguration("reference_map_yaml").perform(context))
    evaluation_label = str(LaunchConfiguration("evaluation_label").perform(context))
    if evaluation_label == "__USE_CONFIG__":
        evaluation_label = ""
    evaluation_output_subdir = str(LaunchConfiguration("evaluation_output_subdir").perform(context))
    if evaluation_output_subdir == "__USE_CONFIG__":
        evaluation_output_subdir = ""
    finalize_experiment_arg = str(LaunchConfiguration("finalize_experiment").perform(context))
    finalize_experiment = (
        True
        if finalize_experiment_arg == "__USE_CONFIG__"
        else parse_bool(finalize_experiment_arg, default=True)
    )
    write_eval_metadata_arg = str(LaunchConfiguration("write_evaluation_metadata").perform(context))
    write_evaluation_metadata = (
        True
        if write_eval_metadata_arg == "__USE_CONFIG__"
        else parse_bool(write_eval_metadata_arg, default=True)
    )
    train_world_sdf = str(get_config_value(cfg, "simulation", "train_world", default="world_house.sdf"))
    test_world_sdf = str(get_config_value(cfg, "simulation", "test_world", default="world_house.sdf"))

    # === CZASY ===
    # pipeline.* = wspólne okna dla AI + Robak + Rywak; timing.* = fallback (np. fast_test bez pipeline)
    def _dataset_duration_from_cfg() -> float:
        pipe = get_config_value(cfg, "pipeline", "dataset_collection_sec", default=None)
        if pipe is not None:
            return float(pipe)
        return float(get_config_value(cfg, "timing", "dataset_duration", default=45.0))

    def _eval_duration_from_cfg() -> float:
        pipe = get_config_value(cfg, "pipeline", "evaluation_sec", default=None)
        if pipe is not None:
            return float(pipe)
        return float(get_config_value(cfg, "timing", "eval_duration", default=60.0))

    _launch_eval = LaunchConfiguration("eval_duration_sec").perform(context)
    if _launch_eval == "__USE_CONFIG__":
        eval_duration_sec = _eval_duration_from_cfg()
    else:
        eval_duration_sec = float(_launch_eval)

    _launch_ds = LaunchConfiguration("dataset_duration_sec").perform(context)
    if _launch_ds == "__USE_CONFIG__":
        dataset_duration_sec = _dataset_duration_from_cfg()
    else:
        dataset_duration_sec = float(_launch_ds)

    # 0 = tylko skalowanie dynamiczne (max(0, faza)) w lifecycle; >0 = dodatkowy dolny limit
    dataset_wait_timeout = float(get_config_value(cfg, "timing", "dataset_wait_timeout", default=0.0))
    bridge_delay = float(get_config_value(cfg, "timing", "bridge_delay", default=3.0))
    spawn_delay = float(get_config_value(cfg, "timing", "spawn_delay", default=5.0))
    slam_configure_delay_cfg = float(get_config_value(cfg, "timing", "slam_configure_delay", default=2.0))
    # SLAM configure przed startem bridge powoduje niestabilny punkt startu mapy.
    slam_configure_delay = max(slam_configure_delay_cfg, bridge_delay + 0.5)
    if slam_configure_delay > slam_configure_delay_cfg + 1e-9:
        print(
            "[INFO] timing.slam_configure_delay podbite do "
            f"{slam_configure_delay:.2f}s (z {slam_configure_delay_cfg:.2f}s), "
            f"bo bridge_delay={bridge_delay:.2f}s."
        )

    driver_start_delay_cfg = get_config_value(cfg, "timing", "driver_start_delay", default=None)
    if driver_start_delay_cfg is None or float(driver_start_delay_cfg) <= 0.0:
        # Domyślnie uruchom driver po bridge i po aktywacji lifecycle SLAM.
        driver_start_delay = max(bridge_delay + 1.0, slam_configure_delay + 0.5)
    else:
        driver_start_delay = float(driver_start_delay_cfg)
    driver_start_delay = max(0.0, driver_start_delay)
    
    # === TRENING ===
    max_epochs = int(get_config_value(cfg, "training", "max_epochs", default=200))
    patience = int(get_config_value(cfg, "training", "patience", default=20))
    min_delta = float(get_config_value(cfg, "training", "min_delta", default=1e-5))
    learning_rate = float(get_config_value(cfg, "training", "learning_rate", default=0.001))
    batch_size = int(get_config_value(cfg, "training", "batch_size", default=128))
    validation_ratio = float(get_config_value(cfg, "training", "validation_ratio", default=0.2))
    split_strategy = str(get_config_value(cfg, "training", "split_strategy", default="tail_holdout_no_shuffle"))
    torch_deterministic = parse_bool(
        get_config_value(cfg, "training", "torch_deterministic", default=False),
        default=False,
    )
    skip_if_model_exists = parse_bool(
        get_config_value(cfg, "training", "skip_if_model_exists", default=True),
        default=True,
    )
    
    # === DATASET ===
    # 0 = wyłącza limit próbek — zatrzymanie po duration_sec (patrz dataset_recorder*.py)
    dataset_max_samples = int(get_config_value(cfg, "dataset", "max_samples", default=0))
    dataset_scan_topic = str(get_config_value(cfg, "dataset", "scan_topic", default="/scan"))
    dataset_odom_topic = str(get_config_value(cfg, "dataset", "odom_topic", default="/odom"))
    dataset_gt_topic = str(get_config_value(cfg, "dataset", "gt_topic", default="/ground_truth_pose"))
    dataset_sync_tolerance_sec = float(get_config_value(cfg, "dataset", "sync_tolerance_sec", default=0.08))
    dataset_sync_pair_gap_sec = float(get_config_value(cfg, "dataset", "sync_pair_gap_sec", default=0.2))
    dataset_interpolate_odom = parse_bool(get_config_value(cfg, "dataset", "interpolate_odom", default=True), default=True)
    dataset_interpolate_gt = parse_bool(get_config_value(cfg, "dataset", "interpolate_gt", default=True), default=True)
    dataset_stop_on_planned_done = parse_bool(
        get_config_value(cfg, "dataset", "stop_on_planned_path_done", default=False),
        default=False,
    )
    dataset_planned_done_topic = str(
        get_config_value(cfg, "dataset", "planned_path_done_topic", default="/planned_path_done")
    )
    dataset_planned_done_min_elapsed_sec = float(
        get_config_value(cfg, "dataset", "planned_path_done_min_elapsed_sec", default=0.0)
    )
    dataset_motion_watchdog_enabled = parse_bool(
        get_config_value(cfg, "dataset", "motion_stall_watchdog_enabled", default=False),
        default=False,
    )
    dataset_motion_watchdog_pose_topic = str(
        get_config_value(cfg, "dataset", "motion_stall_pose_topic", default=dataset_gt_topic)
    )
    dataset_motion_watchdog_min_delta_m = float(
        get_config_value(cfg, "dataset", "motion_stall_min_delta_m", default=0.035)
    )
    dataset_motion_watchdog_timeout_sec = float(
        get_config_value(cfg, "dataset", "motion_stall_timeout_sec", default=35.0)
    )
    dataset_motion_watchdog_startup_grace_sec = float(
        get_config_value(cfg, "dataset", "motion_stall_startup_grace_sec", default=18.0)
    )
    dataset_motion_watchdog_no_pose_timeout_sec = float(
        get_config_value(cfg, "dataset", "motion_stall_no_pose_timeout_sec", default=20.0)
    )
    dataset_motion_watchdog_check_hz = float(
        get_config_value(cfg, "dataset", "motion_stall_check_hz", default=4.0)
    )
    dataset_motion_watchdog_enable_window_guard = parse_bool(
        get_config_value(cfg, "dataset", "motion_stall_enable_window_guard", default=True),
        default=True,
    )
    dataset_motion_watchdog_min_window_progress_m = float(
        get_config_value(cfg, "dataset", "motion_stall_min_window_progress_m", default=0.12)
    )
    dataset_motion_watchdog_window_span_ratio = float(
        get_config_value(cfg, "dataset", "motion_stall_window_span_ratio", default=1.8)
    )
    dataset_motion_watchdog_enable_circling_guard = parse_bool(
        get_config_value(cfg, "dataset", "motion_stall_enable_circling_guard", default=True),
        default=True,
    )
    dataset_motion_watchdog_circling_min_window_path_m = float(
        get_config_value(cfg, "dataset", "motion_stall_circling_min_window_path_m", default=1.6)
    )
    dataset_motion_watchdog_circling_max_net_path_ratio = float(
        get_config_value(cfg, "dataset", "motion_stall_circling_max_net_path_ratio", default=0.25)
    )
    dataset_motion_watchdog_circling_max_net_m = float(
        get_config_value(cfg, "dataset", "motion_stall_circling_max_net_m", default=1.2)
    )
    dataset_motion_watchdog_circling_max_span_m = float(
        get_config_value(cfg, "dataset", "motion_stall_circling_max_span_m", default=2.5)
    )
    gt_cfg = get_config_value(cfg, "ground_truth", default={})
    gt_use_tf_world = parse_bool(gt_cfg.get("use_tf_world", True), default=True)
    gt_tf_world_topic = str(gt_cfg.get("tf_world_topic", "/tf_world"))
    gt_tf_world_timeout = float(gt_cfg.get("tf_world_timeout_sec", 0.5))
    gt_model_name_hint = str(gt_cfg.get("model_name_hint", "diffbot"))
    gt_base_link_hint = str(gt_cfg.get("base_link_hint", "base_link"))
    gt_world_frame_hint = str(gt_cfg.get("world_frame_hint", "world"))
    gt_use_gz_pose_info = parse_bool(gt_cfg.get("use_gz_pose_info", True), default=True)
    gt_gz_pose_info_topic = str(gt_cfg.get("gz_pose_info_topic", ""))
    gt_gz_pose_entity_hint = str(gt_cfg.get("gz_pose_entity_hint", gt_model_name_hint))
    gt_publish_odom_fallback = parse_bool(gt_cfg.get("publish_odom_fallback", False), default=False)
    gt_restamp_output_to_now = parse_bool(gt_cfg.get("restamp_output_to_now", True), default=True)
    gt_propagate_tf_world_with_odom = parse_bool(
        gt_cfg.get("propagate_tf_world_with_odom", True),
        default=True,
    )
    gt_heuristic_max_score = float(gt_cfg.get("heuristic_max_score", 12.0))
    gt_heuristic_bootstrap_max_score = float(gt_cfg.get("heuristic_bootstrap_max_score", 64.0))
    gt_heuristic_max_step = float(gt_cfg.get("heuristic_max_step_m", 0.8))
    gt_ignore_tf_world_after_gz_pose = parse_bool(
        gt_cfg.get("ignore_tf_world_after_gz_pose", True),
        default=True,
    )
    gt_debug_every_n = int(gt_cfg.get("debug_every_n", 0))
    
    # === INFERENCE ===
    # 0 = tylko skalowanie z eval_duration_sec w lifecycle (max(0, 2*eval+60))
    model_wait_timeout = float(get_config_value(cfg, "inference", "model_wait_timeout", default=0.0))
    infer_scan_topic = get_config_value(cfg, "inference", "scan_topic", default="/scan_slam_ai")
    infer_odom_topic = get_config_value(cfg, "inference", "odom_topic", default="/odom")

    # UWAGA: mapujemy nazwy z YAML -> nazwy parametrów w infer_node.py
    infer_pose_topic = get_config_value(cfg, "inference", "output_pose_topic", default="/pose_ai")
    infer_odom_ai_topic = get_config_value(cfg, "inference", "output_odom_topic", default="/odom_ai")
    infer_tf_parent = get_config_value(cfg, "inference", "tf_parent_frame", default="odom_ai")
    infer_tf_child  = get_config_value(cfg, "inference", "tf_child_frame", default="base_link_ai")
    infer_max_correction_trans = float(get_config_value(cfg, "inference", "max_correction_trans", default=0.0))
    infer_max_correction_yaw = float(get_config_value(cfg, "inference", "max_correction_yaw", default=0.0))

    # === ODOMETRY ===
    rw_sigma_xy = float(get_config_value(cfg, "odometry", "rw_sigma_xy", default=0.005))
    rw_sigma_theta = float(get_config_value(cfg, "odometry", "rw_sigma_theta", default=0.003))
    odom_in_topic = get_config_value(cfg, "odometry", "input_topic", default="/odom_raw")
    odom_out_topic = get_config_value(cfg, "odometry", "output_topic", default="/odom")
    odom_frame_id = get_config_value(cfg, "odometry", "frame_id", default="odom")
    odom_child_frame_id = get_config_value(cfg, "odometry", "child_frame_id", default="base_link")

    # === DRIVER ===
    driver_linear_vel = float(get_config_value(cfg, "driver", "linear_velocity", default=0.3))
    driver_angular_vel = float(get_config_value(cfg, "driver", "angular_velocity", default=0.5))
    driver_turn_prob = float(get_config_value(cfg, "driver", "turn_probability", default=0.02))
    driver_obstacle_thresh = float(get_config_value(cfg, "driver", "obstacle_threshold", default=0.5))
    driver_side_thresh = float(get_config_value(cfg, "driver", "side_threshold", default=0.35))
    driver_emergency_thresh = float(get_config_value(cfg, "driver", "emergency_threshold", default=0.25))
    driver_explore_interval = int(get_config_value(cfg, "driver", "explore_interval_ticks", default=30))
    driver_explore_prob = float(get_config_value(cfg, "driver", "explore_turn_probability", default=-1.0))
    driver_door_prob = float(get_config_value(cfg, "driver", "doorway_turn_probability", default=0.0))
    driver_door_open = float(get_config_value(cfg, "driver", "doorway_opening_threshold", default=1.8))
    driver_door_wall = float(get_config_value(cfg, "driver", "doorway_wall_threshold", default=0.9))
    driver_door_min = float(get_config_value(cfg, "driver", "doorway_turn_min_sec", default=0.7))
    driver_door_max = float(get_config_value(cfg, "driver", "doorway_turn_max_sec", default=1.4))
    driver_motion_profile_enabled = parse_bool(
        get_config_value(cfg, "driver", "motion_profile_enabled", default=False),
        default=False,
    )
    driver_linear_vel_min = float(get_config_value(cfg, "driver", "linear_velocity_min", default=driver_linear_vel))
    driver_linear_vel_max = float(get_config_value(cfg, "driver", "linear_velocity_max", default=driver_linear_vel))
    driver_angular_vel_min = float(get_config_value(cfg, "driver", "angular_velocity_min", default=driver_angular_vel))
    driver_angular_vel_max = float(get_config_value(cfg, "driver", "angular_velocity_max", default=driver_angular_vel))
    driver_reverse_probability = float(get_config_value(cfg, "driver", "reverse_probability", default=0.0))
    driver_reverse_speed_min = float(get_config_value(cfg, "driver", "reverse_speed_min", default=0.0))
    driver_reverse_speed_max = float(get_config_value(cfg, "driver", "reverse_speed_max", default=0.0))
    driver_profile_change_interval = float(
        get_config_value(cfg, "driver", "profile_change_interval_sec", default=2.5)
    )
    driver_profile_arc_probability = float(
        get_config_value(cfg, "driver", "profile_arc_probability", default=0.35)
    )
    driver_profile_arc_fraction_min = float(
        get_config_value(cfg, "driver", "profile_arc_fraction_min", default=0.12)
    )
    driver_profile_arc_fraction_max = float(
        get_config_value(cfg, "driver", "profile_arc_fraction_max", default=0.45)
    )
    driver_explore_spin_probability = float(
        get_config_value(cfg, "driver", "explore_spin_probability", default=0.18)
    )
    driver_explore_spin_min = float(
        get_config_value(cfg, "driver", "explore_spin_min_sec", default=1.0)
    )
    driver_explore_spin_max = float(
        get_config_value(cfg, "driver", "explore_spin_max_sec", default=2.4)
    )
    driver_forward_slowdown_min_factor = float(
        get_config_value(cfg, "driver", "forward_slowdown_min_factor", default=0.45)
    )
    driver_nav_sector_deg = float(get_config_value(cfg, "driver", "nav_sector_deg", default=110.0))
    driver_nav_gap_half_window_deg = float(
        get_config_value(cfg, "driver", "nav_gap_half_window_deg", default=16.0)
    )
    driver_nav_safe_clearance = float(
        get_config_value(cfg, "driver", "nav_safe_clearance", default=0.52)
    )
    driver_nav_lookahead_cap = float(
        get_config_value(cfg, "driver", "nav_lookahead_cap", default=4.0)
    )
    driver_nav_heading_gain = float(
        get_config_value(cfg, "driver", "nav_heading_gain", default=1.8)
    )
    driver_nav_avoid_gain = float(
        get_config_value(cfg, "driver", "nav_avoid_gain", default=0.7)
    )
    driver_nav_min_linear_speed = float(
        get_config_value(cfg, "driver", "nav_min_linear_speed", default=0.06)
    )
    driver_nav_heading_bias_max_deg = float(
        get_config_value(cfg, "driver", "nav_heading_bias_max_deg", default=70.0)
    )
    driver_nav_heading_bias_hold_sec = float(
        get_config_value(cfg, "driver", "nav_heading_bias_hold_sec", default=5.0)
    )
    driver_nav_heading_smooth_alpha = float(
        get_config_value(cfg, "driver", "nav_heading_smooth_alpha", default=0.55)
    )
    driver_nav_novelty_lookahead_m = float(
        get_config_value(cfg, "driver", "nav_novelty_lookahead_m", default=1.4)
    )
    driver_nav_novelty_bonus = float(
        get_config_value(cfg, "driver", "nav_novelty_bonus", default=0.85)
    )
    driver_nav_recent_cell_penalty = float(
        get_config_value(cfg, "driver", "nav_recent_cell_penalty", default=1.15)
    )
    driver_robot_front_extent = float(
        get_config_value(cfg, "driver", "robot_front_extent", default=0.15)
    )
    driver_robot_rear_extent = float(
        get_config_value(cfg, "driver", "robot_rear_extent", default=0.15)
    )
    driver_robot_half_width = float(
        get_config_value(cfg, "driver", "robot_half_width", default=0.10)
    )
    driver_robot_safety_margin = float(
        get_config_value(cfg, "driver", "robot_safety_margin", default=0.06)
    )
    driver_repeat_cell_size_m = float(
        get_config_value(cfg, "driver", "repeat_cell_size_m", default=0.9)
    )
    driver_repeat_window_size = int(
        get_config_value(cfg, "driver", "repeat_window_size", default=40)
    )
    driver_repeat_unique_ratio_threshold = float(
        get_config_value(cfg, "driver", "repeat_unique_ratio_threshold", default=0.55)
    )
    driver_repeat_escape_trigger = int(
        get_config_value(cfg, "driver", "repeat_escape_trigger", default=6)
    )
    driver_repeat_escape_turn_sec = float(
        get_config_value(cfg, "driver", "repeat_escape_turn_sec", default=2.8)
    )
    driver_repeat_escape_heading_deg = float(
        get_config_value(cfg, "driver", "repeat_escape_heading_deg", default=85.0)
    )
    driver_debug = parse_bool(get_config_value(cfg, "driver", "debug", default=True), default=True)
    driver_debug_every_n = int(get_config_value(cfg, "driver", "debug_every_n", default=10))
    driver_trajectory_mode_cfg = str(get_config_value(cfg, "driver", "trajectory_mode", default="auto"))
    driver_fixed_linear_velocity = float(
        get_config_value(cfg, "driver", "fixed_linear_velocity", default=0.0)
    )
    driver_fixed_angular_velocity = float(
        get_config_value(cfg, "driver", "fixed_angular_velocity", default=0.0)
    )
    driver_fixed_turn_direction = int(
        get_config_value(cfg, "driver", "fixed_turn_direction", default=1)
    )
    driver_fixed_turn_angle_deg = float(
        get_config_value(cfg, "driver", "fixed_turn_angle_deg", default=90.0)
    )
    driver_no_cycle_straight_base_sec = driver_cfg_float(
        cfg, "no_cycle_straight_base_sec", "fixed_no_cycle_straight_base_sec", 1.2
    )
    driver_no_cycle_straight_step_sec = driver_cfg_float(
        cfg, "no_cycle_straight_step_sec", "fixed_no_cycle_straight_step_sec", 0.55
    )
    driver_no_cycle_levels = driver_cfg_int(
        cfg, "no_cycle_levels", "fixed_no_cycle_levels", 14
    )
    driver_cycle_straight_sec = driver_cfg_float(
        cfg, "cycle_straight_sec", "fixed_cycle_straight_sec", 3.0
    )
    driver_fixed_obstacle_avoidance = parse_bool(
        get_config_value(cfg, "driver", "fixed_obstacle_avoidance", default=True),
        default=True,
    )

    # === ROBAK (ALSAI) ===
    robak_cfg = get_config_value(cfg, "robak", default={})
    robak_dataset_name = str(robak_cfg.get("dataset_name", "dataset_robak.npz"))
    robak_model_name = str(robak_cfg.get("model_name", "model_robak.pt"))
    robak_history_name = str(robak_cfg.get("history_name", "train_history_robak.json"))
    robak_dataset_duration = float(robak_cfg.get("dataset_duration", dataset_duration_sec))
    robak_max_samples = int(robak_cfg.get("max_samples", dataset_max_samples))
    # Obsługa obu nazw kluczy (stare: offset_steps/max_delta_*, nowe: offsets/max_pair_*)
    robak_offsets = list(robak_cfg.get("offsets", robak_cfg.get("offset_steps", [1, 2, 3, 4, 5, 8, 10])))
    robak_min_pair_dist = float(robak_cfg.get("min_pair_dist", 0.0))
    robak_min_pair_dyaw = float(robak_cfg.get("min_pair_dyaw", 0.0))
    robak_min_pair_dt_sec = float(robak_cfg.get("min_pair_dt_sec", 0.0))
    robak_pair_filter_mode = str(robak_cfg.get("pair_filter_mode", "any"))
    robak_max_pair_dist = float(robak_cfg.get("max_pair_dist", robak_cfg.get("max_delta_dist", 1.0)))
    robak_max_pair_dyaw = float(robak_cfg.get("max_pair_dyaw", robak_cfg.get("max_delta_yaw", math.pi)))
    robak_trajectory_mode = normalize_dataset_trajectory_mode(
        robak_cfg.get("trajectory_mode", "any")
    )
    robak_trajectory_cell_size_m = float(robak_cfg.get("trajectory_cell_size_m", 0.20))
    robak_cycle_min_repeat_hits = int(robak_cfg.get("cycle_min_repeat_hits", 1))
    robak_balance_histograms = parse_bool(robak_cfg.get("balance_histograms", True), default=True)
    robak_balance_bins = int(robak_cfg.get("balance_bins", 24))
    robak_balance_translation_use_abs = parse_bool(
        robak_cfg.get("balance_translation_use_abs", False), default=False
    )
    robak_balance_rotation_use_abs = parse_bool(
        robak_cfg.get("balance_rotation_use_abs", True), default=True
    )
    robak_balance_translation_hist_min_m = float(robak_cfg.get("balance_translation_hist_min_m", 0.0))
    robak_balance_translation_hist_max_m = float(robak_cfg.get("balance_translation_hist_max_m", 1.0))
    robak_balance_rotation_hist_min_deg = float(robak_cfg.get("balance_rotation_hist_min_deg", 0.0))
    robak_balance_rotation_hist_max_deg = float(robak_cfg.get("balance_rotation_hist_max_deg", 180.0))
    robak_balance_target_quantile = float(robak_cfg.get("balance_target_quantile", 0.35))
    robak_balance_target_min_per_bin = int(robak_cfg.get("balance_target_min_per_bin", 8))
    robak_balance_upsample_sparse_bins = parse_bool(
        robak_cfg.get("balance_upsample_sparse_bins", True), default=True
    )
    robak_balance_merge_strategy = normalize_balance_merge_strategy(
        str(robak_cfg.get("balance_merge_strategy", "union_unique"))
    )
    robak_save_balanced_component_datasets = parse_bool(
        robak_cfg.get("save_balanced_component_datasets", True), default=True
    )
    robak_balanced_translation_dataset_name = str(
        robak_cfg.get("balanced_translation_dataset_name", "dataset_robak_translation_balanced.npz")
    )
    robak_balanced_rotation_dataset_name = str(
        robak_cfg.get("balanced_rotation_dataset_name", "dataset_robak_rotation_balanced.npz")
    )
    robak_label_frame = str(robak_cfg.get("label_frame", "world"))
    robak_sync_tolerance = float(robak_cfg.get("sync_tolerance_sec", 0.08))
    robak_sync_pair_gap = float(robak_cfg.get("sync_pair_gap_sec", dataset_sync_pair_gap_sec))
    robak_interpolate_gt = parse_bool(robak_cfg.get("interpolate_gt", True), default=True)
    robak_aug_noise_std_scale = float(robak_cfg.get("augment_noise_std_scale", 0.1))
    robak_aug_cut_fraction = float(robak_cfg.get("augment_cut_fraction", 0.1))
    robak_aug_cut_max_points = int(robak_cfg.get("augment_cut_max_points", 20))
    robak_infer_max_step_trans = float(robak_cfg.get("infer_max_step_trans", 0.10))
    robak_infer_max_step_yaw = float(robak_cfg.get("infer_max_step_yaw", 0.30))
    robak_infer_delta_ema_alpha = float(robak_cfg.get("infer_delta_ema_alpha", 0.65))
    robak_infer_odom_heading_alpha = float(robak_cfg.get("infer_odom_heading_alpha", 0.30))
    robak_infer_odom_heading_gain = float(robak_cfg.get("infer_odom_heading_gain", 0.75))
    robak_infer_odom_sync_tolerance = float(robak_cfg.get("infer_odom_sync_tolerance_sec", 0.08))
    robak_infer_odom_delta_xy_alpha = float(robak_cfg.get("infer_odom_delta_xy_alpha", 0.55))
    robak_infer_odom_delta_xy_gain = float(robak_cfg.get("infer_odom_delta_xy_gain", 0.55))
    robak_infer_odom_delta_yaw_alpha = float(robak_cfg.get("infer_odom_delta_yaw_alpha", 0.65))
    robak_infer_odom_delta_yaw_gain = float(robak_cfg.get("infer_odom_delta_yaw_gain", 0.45))
    robak_infer_odom_pose_xy_alpha = float(robak_cfg.get("infer_odom_pose_xy_alpha", 0.12))
    robak_infer_odom_pose_xy_gain = float(robak_cfg.get("infer_odom_pose_xy_gain", 0.30))
    robak_infer_odom_pose_xy_alpha_max = float(robak_cfg.get("infer_odom_pose_xy_alpha_max", 1.0))
    robak_odom_guard_enabled = parse_bool(robak_cfg.get("odom_guard_enabled", False), default=False)
    robak_odom_guard_xy_error_m = float(robak_cfg.get("odom_guard_xy_error_m", 0.0))
    robak_odom_guard_xy_anchor_base = float(robak_cfg.get("odom_guard_xy_anchor_base", 0.75))
    robak_odom_guard_xy_anchor_gain = float(robak_cfg.get("odom_guard_xy_anchor_gain", 0.50))
    robak_odom_guard_yaw_error_rad = float(robak_cfg.get("odom_guard_yaw_error_rad", 0.0))
    robak_odom_guard_yaw_anchor_base = float(robak_cfg.get("odom_guard_yaw_anchor_base", 0.70))
    robak_odom_guard_yaw_anchor_gain = float(robak_cfg.get("odom_guard_yaw_anchor_gain", 0.55))
    robak_infer_odom_topic = str(robak_cfg.get("infer_odom_topic", odom_in_topic))
    robak_infer_init_from = str(robak_cfg.get("infer_init_from", "gt")).strip().lower()
    if robak_infer_init_from not in ("gt", "odom", "none"):
        print(
            f"[WARN] Invalid robak.infer_init_from='{robak_infer_init_from}', "
            "falling back to 'gt' (allowed: gt|odom|none)."
        )
        robak_infer_init_from = "gt"
    robak_lr = float(robak_cfg.get("lr", learning_rate))
    robak_epochs = int(robak_cfg.get("max_epochs", max_epochs))
    robak_patience = int(robak_cfg.get("patience", patience))
    robak_val_ratio = float(robak_cfg.get("val_ratio", validation_ratio))
    robak_split_strategy = str(robak_cfg.get("split_strategy", split_strategy))
    robak_batch = int(robak_cfg.get("batch_size", batch_size))
    robak_torch_deterministic = parse_bool(
        robak_cfg.get("torch_deterministic", torch_deterministic),
        default=torch_deterministic,
    )
    robak_pose_topic = str(robak_cfg.get("pose_topic", "/pose_robak"))
    robak_relay_scan_topic = str(robak_cfg.get("relay_scan_topic", ""))
    robak_force_odom_pose = parse_bool(
        robak_cfg.get("force_odom_pose", False),
        default=False,
    )
    robak_odom_rebase_to_local_origin = parse_bool(
        robak_cfg.get("odom_rebase_to_local_origin", False),
        default=False,
    )
    robak_use_odom_corrections = parse_bool(
        robak_cfg.get("use_odom_corrections", True),
        default=True,
    )
    robak_odom_fallback_before_model_ready = parse_bool(
        robak_cfg.get("odom_fallback_before_model_ready", True),
        default=True,
    )
    robak_pose_topic_no_slam = str(robak_cfg.get("pose_topic_no_slam", "/pose_robak_no_slam"))
    robak_tf_parent_no_slam = str(robak_cfg.get("tf_parent_no_slam", "odom_robak_no_slam"))
    robak_tf_child_no_slam = str(robak_cfg.get("tf_child_no_slam", "base_link_robak_no_slam"))
    # Optional per-track overrides for Robak no-SLAM infer node.
    robak_no_slam_cfg = get_config_value(cfg, "robak_no_slam", default={})
    robak_no_slam_infer_max_step_trans = float(
        robak_no_slam_cfg.get("infer_max_step_trans", robak_infer_max_step_trans)
    )
    robak_no_slam_infer_max_step_yaw = float(
        robak_no_slam_cfg.get("infer_max_step_yaw", robak_infer_max_step_yaw)
    )
    robak_no_slam_infer_delta_ema_alpha = float(
        robak_no_slam_cfg.get("infer_delta_ema_alpha", robak_infer_delta_ema_alpha)
    )
    robak_no_slam_infer_odom_heading_alpha = float(
        robak_no_slam_cfg.get("infer_odom_heading_alpha", robak_infer_odom_heading_alpha)
    )
    robak_no_slam_infer_odom_heading_gain = float(
        robak_no_slam_cfg.get("infer_odom_heading_gain", robak_infer_odom_heading_gain)
    )
    robak_no_slam_infer_odom_sync_tolerance = float(
        robak_no_slam_cfg.get("infer_odom_sync_tolerance_sec", robak_infer_odom_sync_tolerance)
    )
    robak_no_slam_infer_odom_delta_xy_alpha = float(
        robak_no_slam_cfg.get("infer_odom_delta_xy_alpha", robak_infer_odom_delta_xy_alpha)
    )
    robak_no_slam_infer_odom_delta_xy_gain = float(
        robak_no_slam_cfg.get("infer_odom_delta_xy_gain", robak_infer_odom_delta_xy_gain)
    )
    robak_no_slam_infer_odom_delta_yaw_alpha = float(
        robak_no_slam_cfg.get("infer_odom_delta_yaw_alpha", robak_infer_odom_delta_yaw_alpha)
    )
    robak_no_slam_infer_odom_delta_yaw_gain = float(
        robak_no_slam_cfg.get("infer_odom_delta_yaw_gain", robak_infer_odom_delta_yaw_gain)
    )
    robak_no_slam_infer_odom_pose_xy_alpha = float(
        robak_no_slam_cfg.get("infer_odom_pose_xy_alpha", robak_infer_odom_pose_xy_alpha)
    )
    robak_no_slam_infer_odom_pose_xy_gain = float(
        robak_no_slam_cfg.get("infer_odom_pose_xy_gain", robak_infer_odom_pose_xy_gain)
    )
    robak_no_slam_infer_odom_pose_xy_alpha_max = float(
        robak_no_slam_cfg.get("infer_odom_pose_xy_alpha_max", robak_infer_odom_pose_xy_alpha_max)
    )
    robak_no_slam_odom_guard_enabled = parse_bool(
        robak_no_slam_cfg.get("odom_guard_enabled", robak_odom_guard_enabled),
        default=robak_odom_guard_enabled,
    )
    robak_no_slam_odom_guard_xy_error_m = float(
        robak_no_slam_cfg.get("odom_guard_xy_error_m", robak_odom_guard_xy_error_m)
    )
    robak_no_slam_odom_guard_xy_anchor_base = float(
        robak_no_slam_cfg.get("odom_guard_xy_anchor_base", robak_odom_guard_xy_anchor_base)
    )
    robak_no_slam_odom_guard_xy_anchor_gain = float(
        robak_no_slam_cfg.get("odom_guard_xy_anchor_gain", robak_odom_guard_xy_anchor_gain)
    )
    robak_no_slam_odom_guard_yaw_error_rad = float(
        robak_no_slam_cfg.get("odom_guard_yaw_error_rad", robak_odom_guard_yaw_error_rad)
    )
    robak_no_slam_odom_guard_yaw_anchor_base = float(
        robak_no_slam_cfg.get("odom_guard_yaw_anchor_base", robak_odom_guard_yaw_anchor_base)
    )
    robak_no_slam_odom_guard_yaw_anchor_gain = float(
        robak_no_slam_cfg.get("odom_guard_yaw_anchor_gain", robak_odom_guard_yaw_anchor_gain)
    )
    robak_no_slam_infer_odom_topic = str(
        robak_no_slam_cfg.get("infer_odom_topic", robak_infer_odom_topic)
    )
    robak_no_slam_force_odom_pose = parse_bool(
        robak_no_slam_cfg.get("force_odom_pose", robak_force_odom_pose),
        default=robak_force_odom_pose,
    )
    robak_no_slam_odom_rebase_to_local_origin = parse_bool(
        robak_no_slam_cfg.get("odom_rebase_to_local_origin", robak_odom_rebase_to_local_origin),
        default=robak_odom_rebase_to_local_origin,
    )
    robak_no_slam_use_odom_corrections = parse_bool(
        robak_no_slam_cfg.get("use_odom_corrections", robak_use_odom_corrections),
        default=robak_use_odom_corrections,
    )
    robak_no_slam_odom_fallback_before_model_ready = parse_bool(
        robak_no_slam_cfg.get(
            "odom_fallback_before_model_ready",
            robak_odom_fallback_before_model_ready,
        ),
        default=robak_odom_fallback_before_model_ready,
    )
    robak_no_slam_infer_init_from = str(
        robak_no_slam_cfg.get("infer_init_from", robak_infer_init_from)
    ).strip().lower()
    if robak_no_slam_infer_init_from not in ("gt", "odom", "none"):
        print(
            f"[WARN] Invalid robak_no_slam.infer_init_from='{robak_no_slam_infer_init_from}', "
            "falling back to Robak infer_init_from."
        )
        robak_no_slam_infer_init_from = robak_infer_init_from

    # === RYWAK ===
    rywak_cfg = get_config_value(cfg, "rywak", default={})
    rywak_dataset_name = str(rywak_cfg.get("dataset_name", "dataset_rywak.npz"))
    rywak_model_name = str(rywak_cfg.get("model_name", "model_rywak.pt"))
    rywak_history_name = str(rywak_cfg.get("history_name", "train_history_rywak.json"))
    rywak_dataset_duration = float(rywak_cfg.get("dataset_duration", dataset_duration_sec))
    rywak_max_samples = int(rywak_cfg.get("max_samples", dataset_max_samples))
    rywak_odom_label_topic = str(rywak_cfg.get("odom_label_topic", "/odom_raw"))
    rywak_gt_topic = str(rywak_cfg.get("gt_topic", dataset_gt_topic))
    rywak_sync_tolerance = float(rywak_cfg.get("sync_tolerance_sec", 0.08))
    rywak_interpolate_odom = parse_bool(rywak_cfg.get("interpolate_odom", False), default=False)
    rywak_interpolate_gt = parse_bool(rywak_cfg.get("interpolate_gt", True), default=True)
    rywak_sync_pair_gap = float(rywak_cfg.get("sync_pair_gap_sec", 0.2))
    rywak_delta_scan_clip = float(rywak_cfg.get("delta_scan_clip", 2.0))
    rywak_min_sample_dist = float(rywak_cfg.get("min_sample_dist", 0.0))
    rywak_min_sample_dyaw = float(rywak_cfg.get("min_sample_dyaw", 0.0))
    rywak_min_sample_dt_sec = float(rywak_cfg.get("min_sample_dt_sec", 0.0))
    rywak_min_delta_scan_rms = float(rywak_cfg.get("min_delta_scan_rms", 0.0))
    rywak_sample_filter_mode = str(rywak_cfg.get("sample_filter_mode", "any"))
    rywak_balance_histograms = parse_bool(
        rywak_cfg.get("balance_histograms", True),
        default=True,
    )
    rywak_balance_bins = int(rywak_cfg.get("balance_bins", 24))
    rywak_balance_linear_use_abs = parse_bool(
        rywak_cfg.get("balance_linear_use_abs", True),
        default=True,
    )
    rywak_balance_angular_use_abs = parse_bool(
        rywak_cfg.get("balance_angular_use_abs", True),
        default=True,
    )
    rywak_balance_linear_hist_min_mps = float(rywak_cfg.get("balance_linear_hist_min_mps", 0.0))
    rywak_balance_linear_hist_max_mps = float(rywak_cfg.get("balance_linear_hist_max_mps", 2.0))
    rywak_balance_angular_hist_min_radps = float(rywak_cfg.get("balance_angular_hist_min_radps", 0.0))
    rywak_balance_angular_hist_max_radps = float(rywak_cfg.get("balance_angular_hist_max_radps", 3.0))
    rywak_balance_target_quantile = float(rywak_cfg.get("balance_target_quantile", 0.35))
    rywak_balance_target_min_per_bin = int(rywak_cfg.get("balance_target_min_per_bin", 8))
    rywak_balance_upsample_sparse_bins = parse_bool(
        rywak_cfg.get("balance_upsample_sparse_bins", True),
        default=True,
    )
    rywak_balance_merge_strategy = normalize_balance_merge_strategy(
        str(rywak_cfg.get("balance_merge_strategy", "union_unique"))
    )
    rywak_save_balanced_component_datasets = parse_bool(
        rywak_cfg.get("save_balanced_component_datasets", True),
        default=True,
    )
    rywak_trajectory_mode = normalize_dataset_trajectory_mode(
        rywak_cfg.get("trajectory_mode", "any")
    )
    rywak_trajectory_cell_size_m = float(rywak_cfg.get("trajectory_cell_size_m", 0.20))
    rywak_cycle_min_repeat_hits = int(rywak_cfg.get("cycle_min_repeat_hits", 1))
    rywak_balanced_linear_dataset_name = str(
        rywak_cfg.get("balanced_linear_dataset_name", "dataset_rywak_linear_balanced.npz")
    )
    rywak_balanced_angular_dataset_name = str(
        rywak_cfg.get("balanced_angular_dataset_name", "dataset_rywak_angular_balanced.npz")
    )
    rywak_hidden_dims = list(rywak_cfg.get("hidden_dims", [192, 96, 48]))
    rywak_dropout = float(rywak_cfg.get("dropout", 0.1))
    rywak_weight_decay = float(rywak_cfg.get("weight_decay", 1e-4))
    rywak_huber_delta = float(rywak_cfg.get("huber_delta", 1.0))
    rywak_input_noise_std = float(rywak_cfg.get("input_noise_std", 0.02))
    rywak_clip_grad_norm = float(rywak_cfg.get("clip_grad_norm", 1.0))
    rywak_loss_dx_weight = float(rywak_cfg.get("loss_dx_weight", 1.0))
    rywak_loss_dy_weight = float(rywak_cfg.get("loss_dy_weight", 1.0))
    rywak_loss_dtheta_weight = float(rywak_cfg.get("loss_dtheta_weight", 1.5))
    rywak_loss_v_weight = float(rywak_cfg.get("loss_v_weight", 1.0))
    rywak_loss_w_weight = float(rywak_cfg.get("loss_w_weight", 1.5))
    rywak_v_clip_abs = float(rywak_cfg.get("v_clip_abs", 0.45))
    rywak_w_clip_abs = float(rywak_cfg.get("w_clip_abs", 1.20))
    rywak_fuse_odom_v_weight = float(rywak_cfg.get("fuse_odom_v_weight", 0.25))
    rywak_fuse_odom_w_weight = float(rywak_cfg.get("fuse_odom_w_weight", 0.55))
    rywak_fuse_odom_v_gain = float(rywak_cfg.get("fuse_odom_v_gain", 0.45))
    rywak_fuse_odom_w_gain = float(rywak_cfg.get("fuse_odom_w_gain", 0.35))
    rywak_vel_ema_alpha = float(rywak_cfg.get("vel_ema_alpha", 0.60))
    rywak_anchor_yaw_to_odom = float(rywak_cfg.get("anchor_yaw_to_odom", 0.35))
    rywak_anchor_yaw_to_odom_gain = float(rywak_cfg.get("anchor_yaw_to_odom_gain", 0.75))
    rywak_anchor_xy_to_odom = float(rywak_cfg.get("anchor_xy_to_odom", 0.0))
    rywak_anchor_xy_to_odom_gain = float(rywak_cfg.get("anchor_xy_to_odom_gain", 0.0))
    rywak_heading_for_xy_odom_weight = float(rywak_cfg.get("heading_for_xy_odom_weight", 0.60))
    rywak_xy_step_odom_weight = float(rywak_cfg.get("xy_step_odom_weight", 0.35))
    rywak_xy_step_odom_gain = float(rywak_cfg.get("xy_step_odom_gain", 0.45))
    rywak_max_integration_dt = float(rywak_cfg.get("max_integration_dt", 0.20))
    rywak_max_step_trans = float(rywak_cfg.get("max_step_trans", 0.0))
    rywak_max_step_yaw = float(rywak_cfg.get("max_step_yaw", 0.0))
    rywak_infer_odom_topic = str(rywak_cfg.get("infer_odom_topic", rywak_odom_label_topic))
    rywak_infer_init_from_odom_topic = str(
        rywak_cfg.get("infer_init_from_odom_topic", odom_in_topic)
    )
    rywak_lr = float(rywak_cfg.get("lr", learning_rate))
    rywak_epochs = int(rywak_cfg.get("max_epochs", max_epochs))
    rywak_patience = int(rywak_cfg.get("patience", patience))
    rywak_val_ratio = float(rywak_cfg.get("val_ratio", validation_ratio))
    rywak_split_strategy = str(rywak_cfg.get("split_strategy", split_strategy))
    rywak_batch = int(rywak_cfg.get("batch_size", batch_size))
    rywak_torch_deterministic = parse_bool(
        rywak_cfg.get("torch_deterministic", torch_deterministic),
        default=torch_deterministic,
    )
    rywak_pose_topic = str(rywak_cfg.get("pose_topic", "/pose_rywak"))
    rywak_relay_scan_topic = str(rywak_cfg.get("relay_scan_topic", ""))
    rywak_force_odom_pose = parse_bool(
        rywak_cfg.get("force_odom_pose", False),
        default=False,
    )
    rywak_odom_rebase_to_local_origin = parse_bool(
        rywak_cfg.get("odom_rebase_to_local_origin", False),
        default=False,
    )
    rywak_use_odom_corrections = parse_bool(
        rywak_cfg.get("use_odom_corrections", True),
        default=True,
    )
    rywak_odom_fallback_before_model_ready = parse_bool(
        rywak_cfg.get("odom_fallback_before_model_ready", True),
        default=True,
    )
    rywak_odom_guard_enabled = parse_bool(
        rywak_cfg.get("odom_guard_enabled", True),
        default=True,
    )
    rywak_odom_guard_fuse_weight = float(rywak_cfg.get("odom_guard_fuse_weight", 0.95))
    rywak_odom_guard_v_abs_diff = float(rywak_cfg.get("odom_guard_v_abs_diff", 0.35))
    rywak_odom_guard_v_rel_diff = float(rywak_cfg.get("odom_guard_v_rel_diff", 0.60))
    rywak_odom_guard_w_abs_diff = float(rywak_cfg.get("odom_guard_w_abs_diff", 0.80))
    rywak_odom_guard_w_rel_diff = float(rywak_cfg.get("odom_guard_w_rel_diff", 0.80))
    rywak_odom_guard_sign_conflict_speed = float(
        rywak_cfg.get("odom_guard_sign_conflict_speed", 0.08)
    )
    rywak_odom_guard_xy_error_m = float(rywak_cfg.get("odom_guard_xy_error_m", 0.45))
    rywak_odom_guard_xy_anchor_base = float(rywak_cfg.get("odom_guard_xy_anchor_base", 0.80))
    rywak_odom_guard_xy_anchor_gain = float(rywak_cfg.get("odom_guard_xy_anchor_gain", 0.70))
    rywak_odom_guard_yaw_error_rad = float(rywak_cfg.get("odom_guard_yaw_error_rad", 0.35))
    rywak_odom_guard_yaw_anchor_base = float(rywak_cfg.get("odom_guard_yaw_anchor_base", 0.75))
    rywak_odom_guard_yaw_anchor_gain = float(rywak_cfg.get("odom_guard_yaw_anchor_gain", 0.50))
    rywak_pose_topic_no_slam = str(rywak_cfg.get("pose_topic_no_slam", "/pose_rywak_no_slam"))
    rywak_tf_parent_no_slam = str(rywak_cfg.get("tf_parent_no_slam", "odom_rywak_no_slam"))
    rywak_tf_child_no_slam = str(rywak_cfg.get("tf_child_no_slam", "base_link_rywak_no_slam"))
    # Optional per-track overrides for Rywak no-SLAM infer node.
    rywak_no_slam_cfg = get_config_value(cfg, "rywak_no_slam", default={})
    rywak_no_slam_fuse_odom_v_weight = float(
        rywak_no_slam_cfg.get("fuse_odom_v_weight", rywak_fuse_odom_v_weight)
    )
    rywak_no_slam_fuse_odom_w_weight = float(
        rywak_no_slam_cfg.get("fuse_odom_w_weight", rywak_fuse_odom_w_weight)
    )
    rywak_no_slam_fuse_odom_v_gain = float(
        rywak_no_slam_cfg.get("fuse_odom_v_gain", rywak_fuse_odom_v_gain)
    )
    rywak_no_slam_fuse_odom_w_gain = float(
        rywak_no_slam_cfg.get("fuse_odom_w_gain", rywak_fuse_odom_w_gain)
    )
    rywak_no_slam_anchor_yaw_to_odom = float(
        rywak_no_slam_cfg.get("anchor_yaw_to_odom", rywak_anchor_yaw_to_odom)
    )
    rywak_no_slam_anchor_yaw_to_odom_gain = float(
        rywak_no_slam_cfg.get("anchor_yaw_to_odom_gain", rywak_anchor_yaw_to_odom_gain)
    )
    rywak_no_slam_anchor_xy_to_odom = float(
        rywak_no_slam_cfg.get("anchor_xy_to_odom", rywak_anchor_xy_to_odom)
    )
    rywak_no_slam_anchor_xy_to_odom_gain = float(
        rywak_no_slam_cfg.get("anchor_xy_to_odom_gain", rywak_anchor_xy_to_odom_gain)
    )
    rywak_no_slam_heading_for_xy_odom_weight = float(
        rywak_no_slam_cfg.get("heading_for_xy_odom_weight", rywak_heading_for_xy_odom_weight)
    )
    rywak_no_slam_xy_step_odom_weight = float(
        rywak_no_slam_cfg.get("xy_step_odom_weight", rywak_xy_step_odom_weight)
    )
    rywak_no_slam_xy_step_odom_gain = float(
        rywak_no_slam_cfg.get("xy_step_odom_gain", rywak_xy_step_odom_gain)
    )
    rywak_no_slam_sync_tolerance = float(
        rywak_no_slam_cfg.get("sync_tolerance_sec", rywak_sync_tolerance)
    )
    rywak_no_slam_interpolate_odom = parse_bool(
        rywak_no_slam_cfg.get("interpolate_odom", rywak_interpolate_odom),
        default=rywak_interpolate_odom,
    )
    rywak_no_slam_sync_pair_gap = float(
        rywak_no_slam_cfg.get("sync_pair_gap_sec", rywak_sync_pair_gap)
    )
    rywak_no_slam_delta_scan_clip = float(
        rywak_no_slam_cfg.get("delta_scan_clip", rywak_delta_scan_clip)
    )
    rywak_no_slam_v_clip_abs = float(
        rywak_no_slam_cfg.get("v_clip_abs", rywak_v_clip_abs)
    )
    rywak_no_slam_w_clip_abs = float(
        rywak_no_slam_cfg.get("w_clip_abs", rywak_w_clip_abs)
    )
    rywak_no_slam_vel_ema_alpha = float(
        rywak_no_slam_cfg.get("vel_ema_alpha", rywak_vel_ema_alpha)
    )
    rywak_no_slam_max_integration_dt = float(
        rywak_no_slam_cfg.get("max_integration_dt", rywak_max_integration_dt)
    )
    rywak_no_slam_max_step_trans = float(
        rywak_no_slam_cfg.get("max_step_trans", rywak_max_step_trans)
    )
    rywak_no_slam_max_step_yaw = float(
        rywak_no_slam_cfg.get("max_step_yaw", rywak_max_step_yaw)
    )
    rywak_no_slam_infer_odom_topic = str(
        rywak_no_slam_cfg.get("infer_odom_topic", rywak_infer_odom_topic)
    )
    rywak_no_slam_force_odom_pose = parse_bool(
        rywak_no_slam_cfg.get("force_odom_pose", rywak_force_odom_pose),
        default=rywak_force_odom_pose,
    )
    rywak_no_slam_odom_rebase_to_local_origin = parse_bool(
        rywak_no_slam_cfg.get("odom_rebase_to_local_origin", rywak_odom_rebase_to_local_origin),
        default=rywak_odom_rebase_to_local_origin,
    )
    rywak_no_slam_use_odom_corrections = parse_bool(
        rywak_no_slam_cfg.get("use_odom_corrections", rywak_use_odom_corrections),
        default=rywak_use_odom_corrections,
    )
    rywak_no_slam_odom_fallback_before_model_ready = parse_bool(
        rywak_no_slam_cfg.get(
            "odom_fallback_before_model_ready",
            rywak_odom_fallback_before_model_ready,
        ),
        default=rywak_odom_fallback_before_model_ready,
    )
    rywak_no_slam_odom_guard_enabled = parse_bool(
        rywak_no_slam_cfg.get("odom_guard_enabled", rywak_odom_guard_enabled),
        default=rywak_odom_guard_enabled,
    )
    rywak_no_slam_odom_guard_fuse_weight = float(
        rywak_no_slam_cfg.get("odom_guard_fuse_weight", rywak_odom_guard_fuse_weight)
    )
    rywak_no_slam_odom_guard_v_abs_diff = float(
        rywak_no_slam_cfg.get("odom_guard_v_abs_diff", rywak_odom_guard_v_abs_diff)
    )
    rywak_no_slam_odom_guard_v_rel_diff = float(
        rywak_no_slam_cfg.get("odom_guard_v_rel_diff", rywak_odom_guard_v_rel_diff)
    )
    rywak_no_slam_odom_guard_w_abs_diff = float(
        rywak_no_slam_cfg.get("odom_guard_w_abs_diff", rywak_odom_guard_w_abs_diff)
    )
    rywak_no_slam_odom_guard_w_rel_diff = float(
        rywak_no_slam_cfg.get("odom_guard_w_rel_diff", rywak_odom_guard_w_rel_diff)
    )
    rywak_no_slam_odom_guard_sign_conflict_speed = float(
        rywak_no_slam_cfg.get(
            "odom_guard_sign_conflict_speed",
            rywak_odom_guard_sign_conflict_speed,
        )
    )
    rywak_no_slam_odom_guard_xy_error_m = float(
        rywak_no_slam_cfg.get("odom_guard_xy_error_m", rywak_odom_guard_xy_error_m)
    )
    rywak_no_slam_odom_guard_xy_anchor_base = float(
        rywak_no_slam_cfg.get("odom_guard_xy_anchor_base", rywak_odom_guard_xy_anchor_base)
    )
    rywak_no_slam_odom_guard_xy_anchor_gain = float(
        rywak_no_slam_cfg.get("odom_guard_xy_anchor_gain", rywak_odom_guard_xy_anchor_gain)
    )
    rywak_no_slam_odom_guard_yaw_error_rad = float(
        rywak_no_slam_cfg.get("odom_guard_yaw_error_rad", rywak_odom_guard_yaw_error_rad)
    )
    rywak_no_slam_odom_guard_yaw_anchor_base = float(
        rywak_no_slam_cfg.get("odom_guard_yaw_anchor_base", rywak_odom_guard_yaw_anchor_base)
    )
    rywak_no_slam_odom_guard_yaw_anchor_gain = float(
        rywak_no_slam_cfg.get("odom_guard_yaw_anchor_gain", rywak_odom_guard_yaw_anchor_gain)
    )
    rywak_no_slam_infer_init_from_odom_topic = str(
        rywak_no_slam_cfg.get("infer_init_from_odom_topic", rywak_infer_init_from_odom_topic)
    )
    # === SLAM TOOLBOX ===
    slam_cfg_root = get_config_value(cfg, "slam", default={})
    slam_common_cfg = get_config_value(cfg, "slam", "common", default={})
    slam_baseline_cfg = get_config_value(cfg, "slam", "baseline", default={})
    slam_ai_cfg = get_config_value(cfg, "slam", "ai", default={})
    slam_robak_cfg = get_config_value(cfg, "slam", "robak", default={})
    slam_rywak_cfg = get_config_value(cfg, "slam", "rywak", default={})

    # Parametry na poziomie "slam" (np. max_laser_range, resolution) też stosujemy do wszystkich wariantów.
    slam_variant_keys = {"common", "baseline", "ai", "robak", "rywak"}
    slam_global_cfg = {}
    if isinstance(slam_cfg_root, dict):
        for key, value in slam_cfg_root.items():
            if key in slam_variant_keys:
                continue
            if isinstance(value, dict):
                continue
            slam_global_cfg[key] = value

    slam_baseline_params = coerce_slam_param_types(
        merge_params(slam_global_cfg, slam_common_cfg, slam_baseline_cfg)
    )
    slam_ai_params = coerce_slam_param_types(
        merge_params(slam_global_cfg, slam_common_cfg, slam_ai_cfg)
    )
    slam_robak_params = coerce_slam_param_types(
        merge_params(slam_global_cfg, slam_common_cfg, slam_robak_cfg)
    )
    slam_rywak_params = coerce_slam_param_types(
        merge_params(slam_global_cfg, slam_common_cfg, slam_rywak_cfg)
    )
    slam_baseline_map_frame = str(slam_baseline_params.get("map_frame", "map"))
    slam_baseline_odom_frame = str(slam_baseline_params.get("odom_frame", "odom"))
    slam_robak_map_frame = str(slam_robak_params.get("map_frame", "map_robak"))
    slam_robak_odom_frame = str(slam_robak_params.get("odom_frame", "odom_robak"))
    slam_rywak_map_frame = str(slam_rywak_params.get("map_frame", "map_rywak"))
    slam_rywak_odom_frame = str(slam_rywak_params.get("odom_frame", "odom_rywak"))
        
    # === OUTPUT ===
    out_dir = str(get_param("out_dir", ["output", "base_dir"], "out"))
    # experiment_id: jeśli nie podano w launch args, generuj automatycznie
    experiment_id_launch = LaunchConfiguration("experiment_id").perform(context)
    if experiment_id_launch and experiment_id_launch != "__USE_CONFIG__":
        experiment_id = experiment_id_launch
    else:
        experiment_id = generate_experiment_id()

    # Opcjonalnie: ładuj modele *.pt z innego podfolderu out/ (np. test-only z nowym experiment_id)
    _msrc_launch = LaunchConfiguration("model_source_experiment_id").perform(context)
    if _msrc_launch == "__USE_CONFIG__":
        model_source_experiment_id = str(
            get_config_value(cfg, "experiment", "model_source_experiment_id", default="")
        ).strip()
    else:
        model_source_experiment_id = str(_msrc_launch).strip()
    # Opcjonalnie: trenuj na datasetach *.npz z innego podfolderu out/<id>/.
    _dsrc_launch = LaunchConfiguration("dataset_source_experiment_id").perform(context)
    if _dsrc_launch == "__USE_CONFIG__":
        dataset_source_experiment_id = str(
            get_config_value(cfg, "experiment", "dataset_source_experiment_id", default="")
        ).strip()
    else:
        dataset_source_experiment_id = str(_dsrc_launch).strip()

    if not model_source_experiment_id and dataset_source_experiment_id and phase in ("full", "test"):
        # Priorytet: jeśli dla bieżącego experiment_id istnieją już wytrenowane modele,
        # test ma używać ich (typowy przepływ phase=full: train -> test).
        current_exp_dir = os.path.join(out_dir, experiment_id)
        current_exp_models = [
            os.path.join(current_exp_dir, "model.pt"),
            os.path.join(current_exp_dir, robak_model_name),
            os.path.join(current_exp_dir, rywak_model_name),
        ]
        if any(os.path.isfile(path) for path in current_exp_models):
            print(
                "[INFO] model_source_experiment_id not set; "
                f"using models from current experiment_id='{experiment_id}'."
            )
        else:
            # Fallback dla phase=test bez lokalnego treningu: modele z dataset_source_experiment_id.
            candidate_dir = os.path.join(out_dir, dataset_source_experiment_id)
            candidate_models = [
                os.path.join(candidate_dir, "model.pt"),
                os.path.join(candidate_dir, robak_model_name),
                os.path.join(candidate_dir, rywak_model_name),
            ]
            if any(os.path.isfile(path) for path in candidate_models):
                model_source_experiment_id = dataset_source_experiment_id
                print(
                    "[INFO] model_source_experiment_id not set; "
                    f"using dataset_source_experiment_id='{dataset_source_experiment_id}' "
                    "for test-time model loading."
                )
    
    # === ŚCIEŻKI ===
    gazebo_share = get_package_share_directory("ai_slam_gazebo")
    desc_share = get_package_share_directory("ai_slam_description")
    bringup_share = get_package_share_directory("ai_slam_bringup")
    eval_share = get_package_share_directory("ai_slam_eval")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    repo_root = os.path.abspath(os.path.join(bringup_share, "..", "..", "..", "..", ".."))
    gazebo_source_share = source_package_dir(repo_root, "ai_slam_gazebo")
    desc_source_share = source_package_dir(repo_root, "ai_slam_description")
    bringup_source_share = source_package_dir(repo_root, "ai_slam_bringup")
    eval_source_share = source_package_dir(repo_root, "ai_slam_eval")
    office_external_root = os.path.join(
        repo_root,
        "niemoje",
        "Dataset-of-Gazebo-Worlds-Models-and-Maps-master",
        "worlds",
        "office",
        "extracted",
    )
    hospital_external_root = os.path.join(repo_root, "niemoje", "aws-robomaker-hospital-world-ros1")
    existing_resource_paths = [p for p in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(os.pathsep) if p]
    external_resource_paths = [
        office_external_root,
        os.path.join(office_external_root, "models"),
        hospital_external_root,
        os.path.join(hospital_external_root, "models"),
        os.path.join(hospital_external_root, "worlds"),
    ]
    gz_resource_paths = []
    for candidate_path in [
        gazebo_source_share,
        os.path.join(gazebo_source_share, "models") if gazebo_source_share else "",
        os.path.join(gazebo_source_share, "media") if gazebo_source_share else "",
        desc_source_share,
        gazebo_share,
        os.path.join(gazebo_share, "models"),
        os.path.join(gazebo_share, "media"),
        desc_share,
        *external_resource_paths,
        *existing_resource_paths,
    ]:
        if not candidate_path or not os.path.isdir(candidate_path):
            continue
        if candidate_path in gz_resource_paths:
            continue
        gz_resource_paths.append(candidate_path)
    gazebo_model_paths = []
    for candidate_path in [
        os.path.join(gazebo_source_share, "models") if gazebo_source_share else "",
        os.path.join(office_external_root, "models"),
        os.path.join(hospital_external_root, "models"),
        *[p for p in os.environ.get("GAZEBO_MODEL_PATH", "").split(os.pathsep) if p],
    ]:
        if not candidate_path or not os.path.isdir(candidate_path):
            continue
        if candidate_path in gazebo_model_paths:
            continue
        gazebo_model_paths.append(candidate_path)

    # World (można podać nazwę .sdf z ai_slam_gazebo/worlds/ lub ścieżkę absolutną)
    # World (launch-arg world_sdf ma pierwszeństwo nad configiem)
    selected_world_sdf = ""
    world_path_cfg = str(get_config_value(cfg, "simulation", "world_path", default=""))
    sim_cfg = get_config_value(cfg, "simulation", default={})

    if world_path_cfg:
        selected_world_sdf = world_path_cfg
        world_path = (
            world_path_cfg
            if os.path.isabs(world_path_cfg)
            else resolve_world_path(world_path_cfg, gazebo_source_share, gazebo_share)
        )
    else:
        # Priorytet: launch-arg/config simulation.world_sdf -> auto wybór wg fazy.
        if world_sdf_arg and world_sdf_arg != "__AUTO__":
            selected_world_sdf = world_sdf_arg
        else:
            if phase in ("train", "dataset"):
                selected_world_sdf = train_world_sdf
            elif phase == "test":
                selected_world_sdf = test_world_sdf
            else:
                # Launch uruchamia jeden świat na cały przebieg; dla full preferuj test_world
                # aby ewaluacja była zgodna z mapą referencyjną testu.
                if train_world_sdf != test_world_sdf:
                    print(
                        "[WARN] phase=full uses a single world for all steps; "
                        f"selecting test_world='{test_world_sdf}' (train_world='{train_world_sdf}')."
                    )
                selected_world_sdf = test_world_sdf

        world_path = resolve_world_path(selected_world_sdf, gazebo_source_share, gazebo_share)
    if not os.path.isfile(world_path):
        raise FileNotFoundError(
            f"Gazebo world not found: '{world_path}'. "
            f"Requested world='{world_path_cfg or world_sdf_arg or test_world_sdf}'."
        )
    world_name = extract_world_name(world_path)
    if not gt_gz_pose_info_topic:
        gt_gz_pose_info_topic = f"/world/{world_name}/dynamic_pose/info"
    spawn_pose = resolve_spawn_pose(sim_cfg, selected_world_sdf, world_path, world_name)

    bridge_cfg = prefer_source_path(gazebo_source_share, gazebo_share, "config", "bridge.yaml")
    model_sdf = prefer_source_path(desc_source_share, desc_share, "models", "diffbot.sdf")
    urdf_path = prefer_source_path(desc_source_share, desc_share, "urdf", "diffbot.urdf")
    slam_params_baseline = prefer_source_path(bringup_source_share, bringup_share, "config", "slam_toolbox_baseline.yaml")
    slam_params_ai = prefer_source_path(bringup_source_share, bringup_share, "config", "slam_toolbox_ai.yaml")
    world_launch_path = build_world_with_embedded_robot(world_path, model_sdf, "diffbot", spawn_pose)
    
    # Reference map
    if reference_map_yaml_arg and reference_map_yaml_arg != "__USE_CONFIG__":
        ref_map_cfg = reference_map_yaml_arg
    else:
        ref_map_cfg = str(get_config_value(cfg, "evaluation", "reference_map_yaml", default=""))
    if ref_map_cfg:
        if os.path.isabs(ref_map_cfg):
            reference_map_yaml = ref_map_cfg
        else:
            candidate_eval = prefer_source_path(eval_source_share, eval_share, "maps", ref_map_cfg)
            candidate_cfg = (
                os.path.join(os.path.dirname(resolved_config_path), ref_map_cfg)
                if resolved_config_path
                else ref_map_cfg
            )
            if os.path.exists(candidate_cfg):
                reference_map_yaml = candidate_cfg
            else:
                reference_map_yaml = candidate_eval
    else:
        reference_map_yaml = prefer_source_path(eval_source_share, eval_share, "maps", "reference_map.yaml")

    pp_cfg = get_config_value(cfg, "driver", "planned_path", default={}) or {}
    pp_cfg = dict(pp_cfg)
    _wo = pp_cfg.get("world_overrides")
    _bw = os.path.basename(selected_world_sdf) if selected_world_sdf else ""
    if isinstance(_wo, dict) and _bw and _bw in _wo:
        _extra = _wo[_bw]
        if isinstance(_extra, dict):
            pp_cfg.update(_extra)
    driver_use_planned_path = parse_bool(
        get_config_value(cfg, "driver", "use_planned_path", default=False),
        default=False,
    )
    planned_spec_rel = str(pp_cfg.get("spec_yaml", "planned_paths/office_example.yaml")).strip()
    planned_spec_path = (
        planned_spec_rel
        if os.path.isabs(planned_spec_rel)
        else prefer_source_path(bringup_source_share, bringup_share, "config", planned_spec_rel)
    )
    ref_plan_cfg = str(pp_cfg.get("reference_map_yaml", "")).strip()
    if ref_plan_cfg:
        planned_ref_map = (
            ref_plan_cfg
            if os.path.isabs(ref_plan_cfg)
            else prefer_source_path(eval_source_share, eval_share, "maps", ref_plan_cfg)
        )
    else:
        planned_ref_map = reference_map_yaml
    planned_pose_topic = str(pp_cfg.get("pose_topic", "/ground_truth_pose"))
    planned_cmd_topic = str(pp_cfg.get("cmd_topic", "/cmd_vel"))
    planned_lookahead_m = float(pp_cfg.get("lookahead_m", 0.35))
    planned_linear_max = float(pp_cfg.get("linear_vel_max", driver_linear_vel))
    planned_angular_max = float(pp_cfg.get("angular_vel_max", driver_angular_vel))
    planned_goal_tol = float(pp_cfg.get("goal_tolerance_m", 0.22))
    planned_loop = parse_bool(pp_cfg.get("loop_path", True), default=True)
    planned_rate_hz = float(pp_cfg.get("rate_hz", 20.0))
    planned_heading_gain = float(pp_cfg.get("heading_gain", 2.2))
    planned_heading_stop_deg = float(pp_cfg.get("heading_stop_deg", 55.0))
    planned_heading_resume_deg = float(pp_cfg.get("heading_resume_deg", 35.0))
    planned_turn_in_place_max_duration_sec = float(
        pp_cfg.get("turn_in_place_max_duration_sec", 0.0)
    )
    planned_turn_in_place_escape_v_ratio = float(
        pp_cfg.get("turn_in_place_escape_v_ratio", 0.0)
    )
    planned_turn_in_place_escape_v_min_mps = float(
        pp_cfg.get("turn_in_place_escape_v_min_mps", 0.0)
    )
    planned_post_turn_forward_boost_sec = float(
        pp_cfg.get("post_turn_forward_boost_sec", 0.0)
    )
    planned_post_turn_forward_min_v_ratio = float(
        pp_cfg.get("post_turn_forward_min_v_ratio", 0.0)
    )
    planned_post_turn_forward_min_v_mps = float(
        pp_cfg.get("post_turn_forward_min_v_mps", 0.0)
    )
    planned_turn_direction_guard_deg = float(pp_cfg.get("turn_direction_guard_deg", 18.0))
    planned_turn_direction_preference = float(pp_cfg.get("turn_direction_preference", 1.0))
    planned_alignment_cos_power = float(pp_cfg.get("alignment_cos_power", 2.0))
    planned_nearest_backtrack_points = int(pp_cfg.get("nearest_backtrack_points", 8))
    planned_nearest_horizon_m = float(pp_cfg.get("nearest_horizon_m", 6.0))
    planned_ignore_passed_points = parse_bool(pp_cfg.get("ignore_passed_points", True), default=True)
    planned_dataset_excitation_enabled = parse_bool(
        pp_cfg.get("dataset_excitation_enabled", False), default=False
    )
    planned_excitation_period_sec = float(pp_cfg.get("excitation_period_sec", 12.0))
    planned_excitation_v_min_scale = float(pp_cfg.get("excitation_v_min_scale", 0.25))
    planned_excitation_v_max_scale = float(pp_cfg.get("excitation_v_max_scale", 1.0))
    planned_excitation_heading_bias_deg = float(pp_cfg.get("excitation_heading_bias_deg", 12.0))
    planned_dense_default = float(pp_cfg.get("dense_step_m", 0.2))
    planned_map_flip_y = parse_bool(pp_cfg.get("map_flip_y", True), default=True)
    planned_inflate_m = float(pp_cfg.get("inflate_robot_m", 0.35))
    planned_use_astar_param = parse_bool(pp_cfg.get("use_astar", True), default=True)
    planned_pub_ref_marker = parse_bool(pp_cfg.get("publish_reference_path_marker", True), default=True)
    planned_pub_dense_marker = parse_bool(pp_cfg.get("publish_dense_path_marker", True), default=True)
    planned_ref_marker_topic = str(pp_cfg.get("reference_path_marker_topic", "/planned_path_reference"))
    planned_dense_marker_topic = str(pp_cfg.get("reference_path_dense_marker_topic", "/planned_path_dense"))
    planned_marker_frame = str(pp_cfg.get("reference_path_marker_frame", "world"))
    planned_publish_done_topic = parse_bool(pp_cfg.get("publish_completion_topic", True), default=True)
    planned_done_topic = str(pp_cfg.get("completion_topic", "/planned_path_done"))

    eval_sync_tolerance = float(get_config_value(cfg, "evaluation", "sync_tolerance_sec", default=0.15))
    eval_maps_rotate_180 = parse_bool(get_config_value(cfg, "evaluation", "maps_rotate_180", default=True), default=True)
    eval_maps_max_cols = int(get_config_value(cfg, "evaluation", "maps_max_cols", default=3))
    eval_warmup_sec = float(get_config_value(cfg, "evaluation", "warmup_sec", default=0.0))
    eval_points_min_translation = float(get_config_value(cfg, "evaluation", "points_min_translation", default=0.0))
    eval_points_min_rotation = float(get_config_value(cfg, "evaluation", "points_min_rotation", default=0.0))
    eval_points_min_time_gap_sec = float(get_config_value(cfg, "evaluation", "points_min_time_gap_sec", default=0.0))
    eval_points_filter_mode = str(get_config_value(cfg, "evaluation", "points_filter_mode", default="any"))
    eval_points_use_probabilities = parse_bool(
        get_config_value(cfg, "evaluation", "points_use_probabilities", default=True),
        default=True,
    )
    eval_points_occ_logodds_hit = float(
        get_config_value(cfg, "evaluation", "points_occ_logodds_hit", default=0.85)
    )
    eval_points_free_logodds_miss = float(
        get_config_value(cfg, "evaluation", "points_free_logodds_miss", default=0.40)
    )
    eval_points_logodds_min = float(
        get_config_value(cfg, "evaluation", "points_logodds_min", default=-4.0)
    )
    eval_points_logodds_max = float(
        get_config_value(cfg, "evaluation", "points_logodds_max", default=4.0)
    )
    eval_gt_jump_filter_enabled = parse_bool(
        get_config_value(cfg, "evaluation", "gt_jump_filter_enabled", default=True),
        default=True,
    )
    eval_gt_jump_filter_max_step_m = float(
        get_config_value(cfg, "evaluation", "gt_jump_filter_max_step_m", default=2.0)
    )
    
    gz_sim_launch_py = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    # === LOG KONFIGURACJI ===
    print("\n" + "="*70)
    print("AI SLAM EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"  Config file: {config_file or 'none (defaults)'}")
    print(f"  Mode: {mode}")
    print(f"  World: {world_path}")
    print(f"  World launch file: {world_launch_path}")
    print(
        "  Spawn: "
        f"x={spawn_pose['x']:.2f}, y={spawn_pose['y']:.2f}, z={spawn_pose['z']:.2f}, yaw={spawn_pose['yaw']:.2f}"
    )
    print(f"  Seed: {seed}")
    print(f"  GUI: {gui}")
    print(f"  Eval duration: {eval_duration_sec}s")
    print(f"  Dataset duration: {dataset_duration_sec}s")
    print(
        "  Startup delays: "
        f"bridge={bridge_delay:.2f}s, "
        f"slam_configure={slam_configure_delay:.2f}s, "
        f"driver_start={driver_start_delay:.2f}s"
    )
    print(
        f"  Driver: {'planned_path (' + planned_spec_path + ')' if driver_use_planned_path else 'auto_driver'}"
    )
    if driver_use_planned_path:
        print(f"  Planned path ref. map: {planned_ref_map}")
        if isinstance(_wo, dict) and _bw and _bw in _wo:
            print(f"  (world_overrides zastosowane dla świata: {_bw})")
    if dataset_motion_watchdog_enabled:
        print(
            "  Dataset motion watchdog: "
            f"topic={dataset_motion_watchdog_pose_topic}, "
            f"min_delta={dataset_motion_watchdog_min_delta_m:.3f}m, "
            f"stall_timeout={dataset_motion_watchdog_timeout_sec:.1f}s, "
            f"startup_grace={dataset_motion_watchdog_startup_grace_sec:.1f}s, "
            f"window_guard={dataset_motion_watchdog_enable_window_guard}, "
            f"window_min_progress={dataset_motion_watchdog_min_window_progress_m:.3f}m, "
            f"circling_guard={dataset_motion_watchdog_enable_circling_guard}, "
            f"circling_min_path={dataset_motion_watchdog_circling_min_window_path_m:.2f}m, "
            f"circling_max_ratio={dataset_motion_watchdog_circling_max_net_path_ratio:.3f}, "
            f"circling_max_net={dataset_motion_watchdog_circling_max_net_m:.2f}m, "
            f"circling_max_span={dataset_motion_watchdog_circling_max_span_m:.2f}m"
        )
    print(f"  Training: max_epochs={max_epochs}, patience={patience}, lr={learning_rate}")
    print(f"  Output: {out_dir}/{experiment_id}")
    if model_source_experiment_id:
        print(f"  Model source (load *.pt from): {out_dir}/{model_source_experiment_id}")
    if dataset_source_experiment_id and phase in ("full", "train"):
        print(
            "  Dataset source (train *.npz from): "
            f"{out_dir}/{dataset_source_experiment_id}"
        )
    print("="*70 + "\n")

    # === URDF ===
    with open(urdf_path, "r", encoding="utf-8") as f:
        robot_description = f.read()

    # === NODES ===
    is_ai_mode = (mode == "ai")
    is_gui = (gui == "true")
    tracks_cfg = get_config_value(cfg, "tracks", default={})

    tor1_baseline_enabled = parse_bool(tracks_cfg.get("tor1_baseline", True), default=True)
    tor2_ai_slam_enabled = parse_bool(tracks_cfg.get("tor2_ai_slam", True), default=True)
    tor3_local_enabled = parse_bool(tracks_cfg.get("tor3_local", True), default=True)
    tor4_bruteforce_enabled = parse_bool(tracks_cfg.get("tor4_bruteforce", False), default=False)
    tor5_robak_enabled = parse_bool(tracks_cfg.get("tor5_robak", False), default=False)
    tor6_rywak_enabled = parse_bool(tracks_cfg.get("tor6_rywak", False), default=False)
    tor7_robak_no_slam_enabled = parse_bool(
        tracks_cfg.get("tor7_robak_no_slam", False),
        default=False,
    )
    tor8_rywak_no_slam_enabled = parse_bool(
        tracks_cfg.get("tor8_rywak_no_slam", False),
        default=False,
    )

    # fazy (zakładam, że zmienną `phase` już wcześniej wyliczysz z get_param)
    do_dataset_phase = is_ai_mode and (phase in ("full", "train", "dataset"))
    do_train_phase = is_ai_mode and (phase in ("full", "train"))
    do_test_phase  = is_ai_mode and (phase in ("full", "test"))
    do_eval_phase  = (phase in ("full", "test"))
    do_train_only = is_ai_mode and (phase == "train")
    # Dataset AI-SLAM można włączyć zarówno torem baseline (tor1), jak i ai_slam (tor2).
    # Dzięki temu preset datasetowy może mieć tor2 wyłączony, bez utraty dataset_recorder.
    ai_dataset_enabled = (tor1_baseline_enabled or tor2_ai_slam_enabled) and do_dataset_phase
    ai_train_enabled = tor2_ai_slam_enabled and do_train_phase
    ai_test_enabled = tor2_ai_slam_enabled and do_test_phase
    # Robak / Rywak: osobne fazy train/test
    robak_dataset_enabled = tor5_robak_enabled and do_dataset_phase
    robak_train_enabled = tor5_robak_enabled and do_train_phase
    robak_test_enabled  = tor5_robak_enabled and do_test_phase
    robak_no_slam_test_enabled = tor7_robak_no_slam_enabled and do_test_phase
    rywak_dataset_enabled = tor6_rywak_enabled and do_dataset_phase
    rywak_train_enabled = tor6_rywak_enabled and do_train_phase
    rywak_test_enabled  = tor6_rywak_enabled and do_test_phase
    rywak_no_slam_test_enabled = tor8_rywak_no_slam_enabled and do_test_phase
    ai_train_dataset_name = "dataset.npz"
    robak_train_dataset_name = robak_dataset_name
    rywak_train_dataset_name = rywak_dataset_name
    if dataset_source_experiment_id and do_train_phase:
        source_dir = os.path.join(out_dir, dataset_source_experiment_id)
        source_ai_dataset = os.path.join(source_dir, "dataset.npz")
        source_robak_dataset = os.path.join(source_dir, os.path.basename(robak_dataset_name))
        source_rywak_dataset = os.path.join(source_dir, os.path.basename(rywak_dataset_name))
        missing = []
        if ai_train_enabled and not os.path.isfile(source_ai_dataset):
            missing.append(source_ai_dataset)
        if robak_train_enabled and not os.path.isfile(source_robak_dataset):
            missing.append(source_robak_dataset)
        if rywak_train_enabled and not os.path.isfile(source_rywak_dataset):
            missing.append(source_rywak_dataset)
        if missing:
            print(
                "[WARN] dataset_source_experiment_id ustawione, ale brakuje plików datasetu; "
                "fallback do lokalnego zbierania:\n  - "
                + "\n  - ".join(missing)
            )
        else:
            ai_train_dataset_name = source_ai_dataset
            robak_train_dataset_name = source_robak_dataset
            rywak_train_dataset_name = source_rywak_dataset
            ai_dataset_enabled = False
            robak_dataset_enabled = False
            rywak_dataset_enabled = False
            print(
                f"[INFO] Training dataset source: out/{dataset_source_experiment_id} "
                "(dataset recorders disabled for this run)."
            )
    skip_simulation_for_external_train = bool(
        phase == "train"
        and dataset_source_experiment_id
        and ai_dataset_enabled is False
        and robak_dataset_enabled is False
        and rywak_dataset_enabled is False
    )
    if skip_simulation_for_external_train:
        print("[INFO] Phase=train + external dataset source: simulator stack disabled (trainers only).")
    # In trainers-only mode there is no /clock, so training nodes must use wall-time.
    train_nodes_use_sim_time = not skip_simulation_for_external_train
    # Izolacja: osobny scan_fix → /scan_slam_robak|rywak (ten sam łańcuch co SLAM danej metody).
    robak_scan_chain_enabled = robak_dataset_enabled or robak_test_enabled or robak_no_slam_test_enabled
    rywak_scan_chain_enabled = rywak_dataset_enabled or rywak_test_enabled or rywak_no_slam_test_enabled
    robak_dataset_scan_topic = "/scan_slam_robak" if robak_scan_chain_enabled else dataset_scan_topic
    rywak_dataset_scan_topic = "/scan_slam_rywak" if rywak_scan_chain_enabled else dataset_scan_topic

    driver_trajectory_mode = normalize_driver_trajectory_mode(driver_trajectory_mode_cfg)
    if driver_trajectory_mode == "auto":
        requested_dataset_modes = []
        if robak_dataset_enabled and robak_trajectory_mode in ("no_cycle", "cycle"):
            requested_dataset_modes.append(robak_trajectory_mode)
        if rywak_dataset_enabled and rywak_trajectory_mode in ("no_cycle", "cycle"):
            requested_dataset_modes.append(rywak_trajectory_mode)
        if requested_dataset_modes and len(set(requested_dataset_modes)) == 1:
            mode = requested_dataset_modes[0]
            driver_trajectory_mode = "no_cycle" if mode == "no_cycle" else "cycle"
        else:
            driver_trajectory_mode = "auto"
    max_dataset_phase_duration = max(
        dataset_duration_sec if ai_dataset_enabled else 0.0,
        robak_dataset_duration if robak_dataset_enabled else 0.0,
        rywak_dataset_duration if rywak_dataset_enabled else 0.0,
    )
    if max_dataset_phase_duration > 0.0:
        # Trainers may start before recorders finish and real-time factor can drop below 1.0.
        # Keep a conservative wall-time buffer to avoid false "Dataset not found" failures.
        effective_dataset_wait_timeout = max(
            float(dataset_wait_timeout),
            float(max_dataset_phase_duration) * 3.0 + 300.0,
        )
    else:
        effective_dataset_wait_timeout = float(dataset_wait_timeout)
    effective_model_wait_timeout = max(
        float(model_wait_timeout),
        float(eval_duration_sec) * 2.0 + 60.0,
    )
    # tracki tylko w test/full (w train oszczędzamy CPU)
    tor3_local_enabled = tor3_local_enabled and do_eval_phase
    tor4_bruteforce_enabled = tor4_bruteforce_enabled and do_eval_phase
    # Gazebo launch
    gz_launch_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch_py),
        launch_arguments={
            "gz_args": f"{world_launch_path} -r -s --headless-rendering",
            # We drive shutdown explicitly from dataset/train/eval completion handlers below.
            # Leaving Gazebo's own on-exit shutdown enabled causes duplicate shutdown events.
            "on_exit_shutdown": "False",
        }.items(),
        condition=IfCondition(str(not is_gui).lower()),
    )
    
    gz_launch_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch_py),
        launch_arguments={
            "gz_args": f"{world_launch_path} -r",
            "on_exit_shutdown": "False",
        }.items(),
        condition=IfCondition(str(is_gui).lower()),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{"config_file": bridge_cfg}],
        output="screen",
        ros_arguments=["--ros-args", "-p", "log_level:=warn"],
    )
    tf_world_gz_topic = f"/world/{world_name}/dynamic_pose/info"
    bridge_tf_world = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[f"{tf_world_gz_topic}@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"],
        remappings=[(tf_world_gz_topic, gt_tf_world_topic)],
        output="screen",
        ros_arguments=["--ros-args", "-p", "log_level:=warn"],
        condition=IfCondition(str(gt_use_tf_world).lower()),
    )

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
        output="screen",
        ros_arguments=["--ros-args", "-p", "log_level:=warn"],
    )

    scan_fix_baseline = Node(
        package="ai_slam_bringup",
        executable="scan_fix",
        name="scan_fix_baseline",
        parameters=[{
            "use_sim_time": True,
            "in_topic": "/scan",
            "out_topic": "/scan_slam",
            "frame_id": "base_link",
        }],
        output="screen",
    )

    scan_fix_ai = Node(
        package="ai_slam_bringup",
        executable="scan_fix",
        name="scan_fix_ai",
        parameters=[{
            "use_sim_time": True,
            "in_topic": "/scan",
            "out_topic": "/scan_slam_ai",
            "frame_id": "base_link_ai",
        }],
        output="screen",
        condition=IfCondition(str(ai_test_enabled).lower()),
    )

    scan_fix_robak = Node(
        package="ai_slam_bringup",
        executable="scan_fix",
        name="scan_fix_robak",
        parameters=[{
            "use_sim_time": True,
            "in_topic": "/scan",
            "out_topic": "/scan_slam_robak",
            "frame_id": "base_link_robak",
        }],
        output="screen",
        condition=IfCondition(str(robak_scan_chain_enabled).lower()),
    )

    scan_fix_rywak = Node(
        package="ai_slam_bringup",
        executable="scan_fix",
        name="scan_fix_rywak",
        parameters=[{
            "use_sim_time": True,
            "in_topic": "/scan",
            "out_topic": "/scan_slam_rywak",
            "frame_id": "base_link_rywak",
        }],
        output="screen",
        condition=IfCondition(str(rywak_scan_chain_enabled).lower()),
    )
    sm3_cfg = get_config_value(cfg, "scan_matcher", "tor3", default={})
    sm4_cfg = get_config_value(cfg, "scan_matcher", "tor4", default={})

    scan_matcher_local = Node(
        package="ai_slam_bringup",
        executable="scan_matcher",
        name="scan_matcher_local",
        parameters=[{
            "use_sim_time": True,
            "method": "localmap",
            "scan_topic": "/scan_slam",
            "pose_topic": "/pose_scanmatch",
            "twist_topic": "/twist_scanmatch",
            "frame_id": "odom",
            "tf_parent": "odom_scanmatch",
            "tf_child": "base_link_scanmatch",
            "publish_tf": True,
            "publish_every_n": 1,
        }, sm3_cfg],
        output="screen",
        condition=IfCondition(str(tor3_local_enabled).lower()),
    )

    scan_matcher_bruteforce = Node(
        package="ai_slam_bringup",
        executable="scan_matcher",
        name="scan_matcher_bruteforce",
        parameters=[{
            "use_sim_time": True,
            "method": "bruteforce",
            "scan_topic": "/scan_slam",
            "pose_topic": "/pose_bruteforce",
            "twist_topic": "/twist_bruteforce",
            "frame_id": "odom",
            "tf_parent": "odom_bruteforce",
            "tf_child": "base_link_bruteforce",
            "publish_tf": True,
            "publish_every_n": 5,   # <- żeby nie zabić CPU
            "bf_range_xy": 0.15,
            "bf_range_th": 0.25,
            "bf_step_xy": 0.01,
            "bf_step_th": 0.01,
        }, sm4_cfg],
        output="screen",
        condition=IfCondition(str(tor4_bruteforce_enabled).lower()),
    )

    gt_pose = Node(
        package="ai_slam_bringup",
        executable="gt_pose_publisher",
        parameters=[{
            "use_sim_time": True,
            "in_topic": odom_in_topic,
            "out_topic": dataset_gt_topic,
            "frame_id": odom_frame_id,
            "use_tf_world": gt_use_tf_world,
            "tf_world_topic": gt_tf_world_topic,
            "use_gz_pose_info": gt_use_gz_pose_info,
            "gz_pose_info_topic": gt_gz_pose_info_topic,
            "gz_pose_entity_hint": gt_gz_pose_entity_hint,
            "tf_world_timeout_sec": gt_tf_world_timeout,
            "publish_odom_fallback": gt_publish_odom_fallback,
            "restamp_output_to_now": gt_restamp_output_to_now,
            "propagate_tf_world_with_odom": gt_propagate_tf_world_with_odom,
            "model_name_hint": gt_model_name_hint,
            "base_link_hint": gt_base_link_hint,
            "world_frame_hint": gt_world_frame_hint,
            "heuristic_max_score": gt_heuristic_max_score,
            "heuristic_bootstrap_max_score": gt_heuristic_bootstrap_max_score,
            "heuristic_max_step_m": gt_heuristic_max_step,
            "ignore_tf_world_after_gz_pose": gt_ignore_tf_world_after_gz_pose,
            "debug_every_n": gt_debug_every_n,
        }],
        output="screen",
    )

    odom_corruptor = Node(
        package="ai_slam_bringup",
        executable="odom_corruptor",
        parameters=[{
            "use_sim_time": True, 
            "seed": seed,
            "rw_sigma_xy": rw_sigma_xy,
            "rw_sigma_theta": rw_sigma_theta,
            "in_topic": odom_in_topic,
            "out_topic": odom_out_topic,
            "frame_id": odom_frame_id,
            "child_frame_id": odom_child_frame_id,
        }],
        output="screen",
    )

    driver_auto = Node(
        package="ai_slam_bringup",
        executable="auto_driver",
        parameters=[{
            "use_sim_time": True, 
            "seed": seed,
            "linear_velocity": driver_linear_vel,
            "angular_velocity": driver_angular_vel,
            "turn_probability": driver_turn_prob,
            "obstacle_threshold": driver_obstacle_thresh,
            "side_threshold": driver_side_thresh,
            "emergency_threshold": driver_emergency_thresh,
            "explore_interval_ticks": driver_explore_interval,
            "explore_turn_probability": driver_explore_prob,
            "doorway_turn_probability": driver_door_prob,
            "doorway_opening_threshold": driver_door_open,
            "doorway_wall_threshold": driver_door_wall,
            "doorway_turn_min_sec": driver_door_min,
            "doorway_turn_max_sec": driver_door_max,
            "motion_profile_enabled": driver_motion_profile_enabled,
            "linear_velocity_min": driver_linear_vel_min,
            "linear_velocity_max": driver_linear_vel_max,
            "angular_velocity_min": driver_angular_vel_min,
            "angular_velocity_max": driver_angular_vel_max,
            "reverse_probability": driver_reverse_probability,
            "reverse_speed_min": driver_reverse_speed_min,
            "reverse_speed_max": driver_reverse_speed_max,
            "profile_change_interval_sec": driver_profile_change_interval,
            "profile_arc_probability": driver_profile_arc_probability,
            "profile_arc_fraction_min": driver_profile_arc_fraction_min,
            "profile_arc_fraction_max": driver_profile_arc_fraction_max,
            "explore_spin_probability": driver_explore_spin_probability,
            "explore_spin_min_sec": driver_explore_spin_min,
            "explore_spin_max_sec": driver_explore_spin_max,
            "forward_slowdown_min_factor": driver_forward_slowdown_min_factor,
            "nav_sector_deg": driver_nav_sector_deg,
            "nav_gap_half_window_deg": driver_nav_gap_half_window_deg,
            "nav_safe_clearance": driver_nav_safe_clearance,
            "nav_lookahead_cap": driver_nav_lookahead_cap,
            "nav_heading_gain": driver_nav_heading_gain,
            "nav_avoid_gain": driver_nav_avoid_gain,
            "nav_min_linear_speed": driver_nav_min_linear_speed,
            "nav_heading_bias_max_deg": driver_nav_heading_bias_max_deg,
            "nav_heading_bias_hold_sec": driver_nav_heading_bias_hold_sec,
            "nav_heading_smooth_alpha": driver_nav_heading_smooth_alpha,
            "nav_novelty_lookahead_m": driver_nav_novelty_lookahead_m,
            "nav_novelty_bonus": driver_nav_novelty_bonus,
            "nav_recent_cell_penalty": driver_nav_recent_cell_penalty,
            "robot_front_extent": driver_robot_front_extent,
            "robot_rear_extent": driver_robot_rear_extent,
            "robot_half_width": driver_robot_half_width,
            "robot_safety_margin": driver_robot_safety_margin,
            "repeat_cell_size_m": driver_repeat_cell_size_m,
            "repeat_window_size": driver_repeat_window_size,
            "repeat_unique_ratio_threshold": driver_repeat_unique_ratio_threshold,
            "repeat_escape_trigger": driver_repeat_escape_trigger,
            "repeat_escape_turn_sec": driver_repeat_escape_turn_sec,
            "repeat_escape_heading_deg": driver_repeat_escape_heading_deg,
            "trajectory_mode": driver_trajectory_mode,
            "fixed_linear_velocity": driver_fixed_linear_velocity,
            "fixed_angular_velocity": driver_fixed_angular_velocity,
            "fixed_turn_direction": driver_fixed_turn_direction,
            "fixed_turn_angle_deg": driver_fixed_turn_angle_deg,
            "no_cycle_straight_base_sec": driver_no_cycle_straight_base_sec,
            "no_cycle_straight_step_sec": driver_no_cycle_straight_step_sec,
            "no_cycle_levels": driver_no_cycle_levels,
            "cycle_straight_sec": driver_cycle_straight_sec,
            "fixed_obstacle_avoidance": driver_fixed_obstacle_avoidance,
            "debug": driver_debug,
            "debug_every_n": driver_debug_every_n,
            "odom_topic": odom_in_topic,  # zwykle /odom_raw
        }],
        output="screen",
        condition=UnlessCondition(str(driver_use_planned_path).lower()),
    )

    driver_planned = Node(
        package="ai_slam_bringup",
        executable="planned_path_driver",
        parameters=[{
            "use_sim_time": True,
            "path_spec_yaml": planned_spec_path,
            "reference_map_yaml": planned_ref_map,
            "use_astar": planned_use_astar_param,
            "map_flip_y": planned_map_flip_y,
            "inflate_robot_m": planned_inflate_m,
            "dense_step_m": planned_dense_default,
            "pose_topic": planned_pose_topic,
            "cmd_topic": planned_cmd_topic,
            "lookahead_m": planned_lookahead_m,
            "linear_vel_max": planned_linear_max,
            "angular_vel_max": planned_angular_max,
            "goal_tolerance_m": planned_goal_tol,
            "loop_path": planned_loop,
            "rate_hz": planned_rate_hz,
            "heading_gain": planned_heading_gain,
            "heading_stop_deg": planned_heading_stop_deg,
            "heading_resume_deg": planned_heading_resume_deg,
            "turn_in_place_max_duration_sec": planned_turn_in_place_max_duration_sec,
            "turn_in_place_escape_v_ratio": planned_turn_in_place_escape_v_ratio,
            "turn_in_place_escape_v_min_mps": planned_turn_in_place_escape_v_min_mps,
            "post_turn_forward_boost_sec": planned_post_turn_forward_boost_sec,
            "post_turn_forward_min_v_ratio": planned_post_turn_forward_min_v_ratio,
            "post_turn_forward_min_v_mps": planned_post_turn_forward_min_v_mps,
            "turn_direction_guard_deg": planned_turn_direction_guard_deg,
            "turn_direction_preference": planned_turn_direction_preference,
            "alignment_cos_power": planned_alignment_cos_power,
            "nearest_backtrack_points": planned_nearest_backtrack_points,
            "nearest_horizon_m": planned_nearest_horizon_m,
            "ignore_passed_points": planned_ignore_passed_points,
            "dataset_excitation_enabled": planned_dataset_excitation_enabled,
            "excitation_period_sec": planned_excitation_period_sec,
            "excitation_v_min_scale": planned_excitation_v_min_scale,
            "excitation_v_max_scale": planned_excitation_v_max_scale,
            "excitation_heading_bias_deg": planned_excitation_heading_bias_deg,
            "publish_completion_topic": planned_publish_done_topic,
            "completion_topic": planned_done_topic,
            "publish_reference_path_marker": planned_pub_ref_marker,
            "publish_dense_path_marker": planned_pub_dense_marker,
            "reference_path_marker_topic": planned_ref_marker_topic,
            "reference_path_dense_marker_topic": planned_dense_marker_topic,
            "reference_path_marker_frame": planned_marker_frame,
        }],
        output="screen",
        condition=IfCondition(str(driver_use_planned_path).lower()),
    )

    # SLAM Toolbox nodes - output="log" to suppress TF_OLD_DATA spam
    slam_baseline = LifecycleNode(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox_baseline",
        namespace="",
        parameters=[slam_params_baseline, {"use_sim_time": True}, slam_baseline_params],
        output="log",  # Redirect to log file instead of screen
        arguments=["--ros-args", "--log-level", "warn"],
        # W fazie dataset SLAM baseline nie jest potrzebny (datasety idą z /scan_slam,/scan_slam_robak,/scan_slam_rywak).
        # Wyłączenie go znacząco poprawia real-time factor.
        condition=IfCondition(str(do_eval_phase).lower()),
    )

    slam_ai = LifecycleNode(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox_ai",
        namespace="",
        parameters=[slam_params_ai, {"use_sim_time": True}, slam_ai_params],
        arguments=["--ros-args", "--log-level", "warn"],
        remappings=[("/map", "/map_ai")],
        output="log",  # Redirect to log file instead of screen
        condition=IfCondition(str(ai_test_enabled).lower()),
    )
    slam_robak = LifecycleNode(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox_robak",
        namespace="",
        parameters=[{"use_sim_time": True}, slam_robak_params],
        arguments=["--ros-args", "--log-level", "warn"],
        remappings=[("/map", "/map_robak")],
        output="log",
        condition=IfCondition(str(robak_test_enabled).lower()),
    )

    slam_rywak = LifecycleNode(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox_rywak",
        namespace="",
        parameters=[{"use_sim_time": True}, slam_rywak_params],
        arguments=["--ros-args", "--log-level", "warn"],
        remappings=[("/map", "/map_rywak")],
        output="log",
        condition=IfCondition(str(rywak_test_enabled).lower()),
    )

    # Lifecycle management
    configure_baseline = TimerAction(
        period=slam_configure_delay,
        actions=[
            EmitEvent(event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_baseline),
                transition_id=Transition.TRANSITION_CONFIGURE
            ))
        ],
        condition=IfCondition(str(do_eval_phase).lower()),
    )
    
    activate_baseline = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_baseline,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                LogInfo(msg="[LifecycleLaunch] slam_toolbox_baseline is activating."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_baseline),
                    transition_id=Transition.TRANSITION_ACTIVATE
                ))
            ]
        ),
        condition=IfCondition(str(do_eval_phase).lower()),
    )

    configure_ai = TimerAction(
        period=slam_configure_delay,
        actions=[
            EmitEvent(event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_ai),
                transition_id=Transition.TRANSITION_CONFIGURE
            ))
        ],
        condition=IfCondition(str(ai_test_enabled).lower()),
    )
    
    activate_ai = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_ai,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                LogInfo(msg="[LifecycleLaunch] slam_toolbox_ai is activating."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_ai),
                    transition_id=Transition.TRANSITION_ACTIVATE
                ))
            ]
        ),
        condition=IfCondition(str(ai_test_enabled).lower()),
    )
    configure_robak = TimerAction(
        period=slam_configure_delay,
        actions=[
            EmitEvent(event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_robak),
                transition_id=Transition.TRANSITION_CONFIGURE
            ))
        ],
        condition=IfCondition(str(robak_test_enabled).lower()),
    )

    activate_robak = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_robak,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                LogInfo(msg="[LifecycleLaunch] slam_toolbox_robak is activating."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_robak),
                    transition_id=Transition.TRANSITION_ACTIVATE
                ))
            ]
        ),
        condition=IfCondition(str(robak_test_enabled).lower()),
    )

    configure_rywak = TimerAction(
        period=slam_configure_delay,
        actions=[
            EmitEvent(event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_rywak),
                transition_id=Transition.TRANSITION_CONFIGURE
            ))
        ],
        condition=IfCondition(str(rywak_test_enabled).lower()),
    )

    activate_rywak = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_rywak,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                LogInfo(msg="[LifecycleLaunch] slam_toolbox_rywak is activating."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_rywak),
                    transition_id=Transition.TRANSITION_ACTIVATE
                ))
            ]
        ),
        condition=IfCondition(str(rywak_test_enabled).lower()),
    )

    # AI Pipeline nodes
    dataset_motion_watchdog = Node(
        package="ai_slam_bringup",
        executable="dataset_motion_watchdog",
        parameters=[{
            "use_sim_time": True,
            "pose_topic": dataset_motion_watchdog_pose_topic,
            "min_motion_delta_m": dataset_motion_watchdog_min_delta_m,
            "stall_timeout_sec": dataset_motion_watchdog_timeout_sec,
            "startup_grace_sec": dataset_motion_watchdog_startup_grace_sec,
            "no_pose_timeout_sec": dataset_motion_watchdog_no_pose_timeout_sec,
            "check_hz": dataset_motion_watchdog_check_hz,
            "enable_window_progress_guard": dataset_motion_watchdog_enable_window_guard,
            "stall_min_window_progress_m": dataset_motion_watchdog_min_window_progress_m,
            "stall_window_span_ratio": dataset_motion_watchdog_window_span_ratio,
            "enable_circling_guard": dataset_motion_watchdog_enable_circling_guard,
            "stall_circling_min_window_path_m": dataset_motion_watchdog_circling_min_window_path_m,
            "stall_circling_max_net_path_ratio": dataset_motion_watchdog_circling_max_net_path_ratio,
            "stall_circling_max_net_m": dataset_motion_watchdog_circling_max_net_m,
            "stall_circling_max_span_m": dataset_motion_watchdog_circling_max_span_m,
            "log_alive_heartbeat": False,
        }],
        output="screen",
        condition=IfCondition(str(do_dataset_phase and dataset_motion_watchdog_enabled).lower()),
    )

    dataset_rec = Node(
        package="ai_slam_ai",
        executable="dataset_recorder",
        parameters=[{
            "use_sim_time": True, 
            "seed": seed, 
            "out_dir": out_dir, 
            "experiment_id": experiment_id,
            "duration_sec": dataset_duration_sec,
            "max_samples": dataset_max_samples,
            "scan_topic": dataset_scan_topic,
            "odom_topic": dataset_odom_topic,
            "gt_topic": dataset_gt_topic,
            "sync_tolerance_sec": dataset_sync_tolerance_sec,
            "sync_pair_gap_sec": dataset_sync_pair_gap_sec,
            "interpolate_odom": dataset_interpolate_odom,
            "interpolate_gt": dataset_interpolate_gt,
            "stop_on_planned_path_done": dataset_stop_on_planned_done,
            "planned_path_done_topic": dataset_planned_done_topic,
            "planned_path_done_min_elapsed_sec": dataset_planned_done_min_elapsed_sec,
        }],
        output="screen",
        condition=IfCondition(str(ai_dataset_enabled).lower()),
    )

    trainer = Node(
        package="ai_slam_ai",
        executable="train_model",
        parameters=[{
            "use_sim_time": train_nodes_use_sim_time,
            "seed": seed, 
            "out_dir": out_dir, 
            "experiment_id": experiment_id,
            "dataset_name": ai_train_dataset_name,
            "skip_if_model_exists": skip_if_model_exists,
            "dataset_wait_timeout": effective_dataset_wait_timeout,
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "lr": learning_rate,
            "batch_size": batch_size,
            "val_ratio": validation_ratio,
            "split_strategy": split_strategy,
            "torch_deterministic": torch_deterministic,
        }],
        output="screen",
        condition=IfCondition(str(ai_train_enabled).lower()),
    )

    infer = Node(
        package="ai_slam_ai",
        executable="infer_node",
        parameters=[{
            "use_sim_time": True, 
            "seed": seed, 
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "model_source_experiment_id": model_source_experiment_id,
            "model_wait_timeout": effective_model_wait_timeout,
            "scan_topic": infer_scan_topic,
            "odom_topic": infer_odom_topic,
            "pose_topic": infer_pose_topic,         # zgodnie z infer_node.py :contentReference[oaicite:18]{index=18}
            "odom_ai_topic": infer_odom_ai_topic,   # zgodnie z infer_node.py :contentReference[oaicite:19]{index=19}
            "tf_parent": infer_tf_parent,           # zgodnie z infer_node.py :contentReference[oaicite:20]{index=20}
            "tf_child": infer_tf_child,             # zgodnie z infer_node.py :contentReference[oaicite:21]{index=21}
            "max_correction_trans": infer_max_correction_trans,
            "max_correction_yaw": infer_max_correction_yaw,
   
        }],
        output="screen",
        condition=IfCondition(str(ai_test_enabled).lower()),
    )
    # --- PORÓWNANIA: Robak (scan-scan -> delta pose) ---
    dataset_rec_robak = Node(
        package="ai_slam_ai",
        executable="dataset_recorder_robak",
        parameters=[{
            "use_sim_time": True,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "duration_sec": robak_dataset_duration,
            "max_samples": robak_max_samples,
            "scan_topic": robak_dataset_scan_topic,
            "gt_topic": dataset_gt_topic,
            "dataset_name": robak_dataset_name,
            "offsets": robak_offsets,
            "min_pair_dist": robak_min_pair_dist,
            "min_pair_dyaw": robak_min_pair_dyaw,
            "min_pair_dt_sec": robak_min_pair_dt_sec,
            "pair_filter_mode": robak_pair_filter_mode,
            "max_pair_dist": robak_max_pair_dist,
            "max_pair_dyaw": robak_max_pair_dyaw,
            "trajectory_mode": robak_trajectory_mode,
            "trajectory_cell_size_m": robak_trajectory_cell_size_m,
            "cycle_min_repeat_hits": robak_cycle_min_repeat_hits,
            "label_frame": robak_label_frame,
            "sync_tolerance_sec": robak_sync_tolerance,
            "sync_pair_gap_sec": robak_sync_pair_gap,
            "interpolate_gt": robak_interpolate_gt,
            "augment_noise_std_scale": robak_aug_noise_std_scale,
            "augment_cut_fraction": robak_aug_cut_fraction,
            "augment_cut_max_points": robak_aug_cut_max_points,
            "balance_histograms": robak_balance_histograms,
            "balance_bins": robak_balance_bins,
            "balance_translation_use_abs": robak_balance_translation_use_abs,
            "balance_rotation_use_abs": robak_balance_rotation_use_abs,
            "balance_translation_hist_min_m": robak_balance_translation_hist_min_m,
            "balance_translation_hist_max_m": robak_balance_translation_hist_max_m,
            "balance_rotation_hist_min_deg": robak_balance_rotation_hist_min_deg,
            "balance_rotation_hist_max_deg": robak_balance_rotation_hist_max_deg,
            "balance_target_quantile": robak_balance_target_quantile,
            "balance_target_min_per_bin": robak_balance_target_min_per_bin,
            "balance_upsample_sparse_bins": robak_balance_upsample_sparse_bins,
            "balance_merge_strategy": robak_balance_merge_strategy,
            "save_balanced_component_datasets": robak_save_balanced_component_datasets,
            "balanced_translation_dataset_name": robak_balanced_translation_dataset_name,
            "balanced_rotation_dataset_name": robak_balanced_rotation_dataset_name,
            "stop_on_planned_path_done": dataset_stop_on_planned_done,
            "planned_path_done_topic": dataset_planned_done_topic,
            "planned_path_done_min_elapsed_sec": dataset_planned_done_min_elapsed_sec,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(robak_dataset_enabled).lower()),
    )

    trainer_robak = Node(
        package="ai_slam_ai",
        executable="train_model_robak",
        parameters=[{
            "use_sim_time": train_nodes_use_sim_time,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "dataset_name": robak_train_dataset_name,
            "model_name": robak_model_name,
            "history_name": robak_history_name,
            "skip_if_model_exists": skip_if_model_exists,
            "dataset_wait_timeout": effective_dataset_wait_timeout,
            "max_epochs": robak_epochs,
            "patience": robak_patience,
            "min_delta": min_delta,
            "lr": robak_lr,
            "batch_size": robak_batch,
            "val_ratio": robak_val_ratio,
            "split_strategy": robak_split_strategy,
            "torch_deterministic": robak_torch_deterministic,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(robak_train_enabled).lower()),
    )

    infer_robak = Node(
        package="ai_slam_ai",
        executable="infer_robak_node",
        parameters=[{
            "use_sim_time": True,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "model_source_experiment_id": model_source_experiment_id,
            "model_name": robak_model_name,
            "scan_topic": "/scan_slam_robak",
            "relay_scan_topic": robak_relay_scan_topic,
            "pose_topic": robak_pose_topic,
            "init_from": robak_infer_init_from,
            "gt_topic": dataset_gt_topic,
            "odom_topic": robak_infer_odom_topic,
            "max_step_trans": robak_infer_max_step_trans,
            "max_step_yaw": robak_infer_max_step_yaw,
            "delta_ema_alpha": robak_infer_delta_ema_alpha,
            "odom_heading_alpha": robak_infer_odom_heading_alpha,
            "odom_heading_gain": robak_infer_odom_heading_gain,
            "odom_sync_tolerance_sec": robak_infer_odom_sync_tolerance,
            "odom_delta_xy_alpha": robak_infer_odom_delta_xy_alpha,
            "odom_delta_xy_gain": robak_infer_odom_delta_xy_gain,
            "odom_delta_yaw_alpha": robak_infer_odom_delta_yaw_alpha,
            "odom_delta_yaw_gain": robak_infer_odom_delta_yaw_gain,
            "odom_pose_xy_alpha": robak_infer_odom_pose_xy_alpha,
            "odom_pose_xy_gain": robak_infer_odom_pose_xy_gain,
            "odom_pose_xy_alpha_max": robak_infer_odom_pose_xy_alpha_max,
            "odom_guard_enabled": robak_odom_guard_enabled,
            "odom_guard_xy_error_m": robak_odom_guard_xy_error_m,
            "odom_guard_xy_anchor_base": robak_odom_guard_xy_anchor_base,
            "odom_guard_xy_anchor_gain": robak_odom_guard_xy_anchor_gain,
            "odom_guard_yaw_error_rad": robak_odom_guard_yaw_error_rad,
            "odom_guard_yaw_anchor_base": robak_odom_guard_yaw_anchor_base,
            "odom_guard_yaw_anchor_gain": robak_odom_guard_yaw_anchor_gain,
            "odom_rebase_to_local_origin": robak_odom_rebase_to_local_origin,
            "use_odom_corrections": robak_use_odom_corrections,
            "force_odom_pose": robak_force_odom_pose,
            "odom_fallback_before_model_ready": robak_odom_fallback_before_model_ready,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(robak_test_enabled).lower()),
    )
    infer_robak_no_slam = Node(
        package="ai_slam_ai",
        executable="infer_robak_node",
        parameters=[{
            "use_sim_time": True,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "model_source_experiment_id": model_source_experiment_id,
            "model_name": robak_model_name,
            # Ten sam tor skanu co Robak+SLAM (scan_fix_robak); różnica to brak mapy SLAM w pętli.
            "scan_topic": "/scan_slam_robak",
            "pose_topic": robak_pose_topic_no_slam,
            "tf_parent": robak_tf_parent_no_slam,
            "tf_child": robak_tf_child_no_slam,
            "init_from": robak_no_slam_infer_init_from,
            "gt_topic": dataset_gt_topic,
            "odom_topic": robak_no_slam_infer_odom_topic,
            "max_step_trans": robak_no_slam_infer_max_step_trans,
            "max_step_yaw": robak_no_slam_infer_max_step_yaw,
            "delta_ema_alpha": robak_no_slam_infer_delta_ema_alpha,
            "odom_heading_alpha": robak_no_slam_infer_odom_heading_alpha,
            "odom_heading_gain": robak_no_slam_infer_odom_heading_gain,
            "odom_sync_tolerance_sec": robak_no_slam_infer_odom_sync_tolerance,
            "odom_delta_xy_alpha": robak_no_slam_infer_odom_delta_xy_alpha,
            "odom_delta_xy_gain": robak_no_slam_infer_odom_delta_xy_gain,
            "odom_delta_yaw_alpha": robak_no_slam_infer_odom_delta_yaw_alpha,
            "odom_delta_yaw_gain": robak_no_slam_infer_odom_delta_yaw_gain,
            "odom_pose_xy_alpha": robak_no_slam_infer_odom_pose_xy_alpha,
            "odom_pose_xy_gain": robak_no_slam_infer_odom_pose_xy_gain,
            "odom_pose_xy_alpha_max": robak_no_slam_infer_odom_pose_xy_alpha_max,
            "odom_guard_enabled": robak_no_slam_odom_guard_enabled,
            "odom_guard_xy_error_m": robak_no_slam_odom_guard_xy_error_m,
            "odom_guard_xy_anchor_base": robak_no_slam_odom_guard_xy_anchor_base,
            "odom_guard_xy_anchor_gain": robak_no_slam_odom_guard_xy_anchor_gain,
            "odom_guard_yaw_error_rad": robak_no_slam_odom_guard_yaw_error_rad,
            "odom_guard_yaw_anchor_base": robak_no_slam_odom_guard_yaw_anchor_base,
            "odom_guard_yaw_anchor_gain": robak_no_slam_odom_guard_yaw_anchor_gain,
            "odom_rebase_to_local_origin": robak_no_slam_odom_rebase_to_local_origin,
            "use_odom_corrections": robak_no_slam_use_odom_corrections,
            "force_odom_pose": robak_no_slam_force_odom_pose,
            "odom_fallback_before_model_ready": robak_no_slam_odom_fallback_before_model_ready,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(robak_no_slam_test_enabled).lower()),
    )

    # --- PORÓWNANIA: Rywak (d_theta1 + d_theta2 + delta_scan -> v,w) ---
    dataset_rec_rywak = Node(
        package="ai_slam_ai",
        executable="dataset_recorder_rywak",
        parameters=[{
            "use_sim_time": True,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "duration_sec": rywak_dataset_duration,
            "max_samples": rywak_max_samples,
            "scan_topic": rywak_dataset_scan_topic,
            "odom_topic": rywak_odom_label_topic,
            "gt_topic": rywak_gt_topic,
            "sync_tolerance_sec": rywak_sync_tolerance,
            "interpolate_odom": rywak_interpolate_odom,
            "interpolate_gt": rywak_interpolate_gt,
            "sync_pair_gap_sec": rywak_sync_pair_gap,
            "delta_scan_clip": rywak_delta_scan_clip,
            "min_sample_dist": rywak_min_sample_dist,
            "min_sample_dyaw": rywak_min_sample_dyaw,
            "min_sample_dt_sec": rywak_min_sample_dt_sec,
            "min_delta_scan_rms": rywak_min_delta_scan_rms,
            "sample_filter_mode": rywak_sample_filter_mode,
            "balance_histograms": rywak_balance_histograms,
            "balance_bins": rywak_balance_bins,
            "balance_linear_use_abs": rywak_balance_linear_use_abs,
            "balance_angular_use_abs": rywak_balance_angular_use_abs,
            "balance_linear_hist_min_mps": rywak_balance_linear_hist_min_mps,
            "balance_linear_hist_max_mps": rywak_balance_linear_hist_max_mps,
            "balance_angular_hist_min_radps": rywak_balance_angular_hist_min_radps,
            "balance_angular_hist_max_radps": rywak_balance_angular_hist_max_radps,
            "balance_target_quantile": rywak_balance_target_quantile,
            "balance_target_min_per_bin": rywak_balance_target_min_per_bin,
            "balance_upsample_sparse_bins": rywak_balance_upsample_sparse_bins,
            "balance_merge_strategy": rywak_balance_merge_strategy,
            "save_balanced_component_datasets": rywak_save_balanced_component_datasets,
            "balanced_linear_dataset_name": rywak_balanced_linear_dataset_name,
            "balanced_angular_dataset_name": rywak_balanced_angular_dataset_name,
            "stop_on_planned_path_done": dataset_stop_on_planned_done,
            "planned_path_done_topic": dataset_planned_done_topic,
            "planned_path_done_min_elapsed_sec": dataset_planned_done_min_elapsed_sec,
            "trajectory_mode": rywak_trajectory_mode,
            "trajectory_cell_size_m": rywak_trajectory_cell_size_m,
            "cycle_min_repeat_hits": rywak_cycle_min_repeat_hits,
            "dataset_name": rywak_dataset_name,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(rywak_dataset_enabled).lower()),
    )

    trainer_rywak = Node(
        package="ai_slam_ai",
        executable="train_model_rywak",
        parameters=[{
            "use_sim_time": train_nodes_use_sim_time,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "dataset_name": rywak_train_dataset_name,
            "model_name": rywak_model_name,
            "history_name": rywak_history_name,
            "skip_if_model_exists": skip_if_model_exists,
            "dataset_wait_timeout": effective_dataset_wait_timeout,
            "max_epochs": rywak_epochs,
            "patience": rywak_patience,
            "min_delta": min_delta,
            "lr": rywak_lr,
            "batch_size": rywak_batch,
            "val_ratio": rywak_val_ratio,
            "split_strategy": rywak_split_strategy,
            "torch_deterministic": rywak_torch_deterministic,
            "hidden_dims": rywak_hidden_dims,
            "dropout": rywak_dropout,
            "weight_decay": rywak_weight_decay,
            "huber_delta": rywak_huber_delta,
            "input_noise_std": rywak_input_noise_std,
            "clip_grad_norm": rywak_clip_grad_norm,
            "loss_dx_weight": rywak_loss_dx_weight,
            "loss_dy_weight": rywak_loss_dy_weight,
            "loss_dtheta_weight": rywak_loss_dtheta_weight,
            "loss_v_weight": rywak_loss_v_weight,
            "loss_w_weight": rywak_loss_w_weight,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(rywak_train_enabled).lower()),
    )

    infer_rywak = Node(
        package="ai_slam_ai",
        executable="infer_rywak_node",
        parameters=[{
            "use_sim_time": True,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "model_source_experiment_id": model_source_experiment_id,
            "model_name": rywak_model_name,
            "scan_topic": "/scan_slam_rywak",
            "relay_scan_topic": rywak_relay_scan_topic,
            "pose_topic": rywak_pose_topic,
            "odom_topic": rywak_infer_odom_topic,
            "init_from_odom_topic": rywak_infer_init_from_odom_topic,
            "sync_tolerance_sec": rywak_sync_tolerance,
            "interpolate_odom": rywak_interpolate_odom,
            "sync_pair_gap_sec": rywak_sync_pair_gap,
            "delta_scan_clip": rywak_delta_scan_clip,
            "v_clip_abs": rywak_v_clip_abs,
            "w_clip_abs": rywak_w_clip_abs,
            "fuse_odom_v_weight": rywak_fuse_odom_v_weight,
            "fuse_odom_w_weight": rywak_fuse_odom_w_weight,
            "fuse_odom_v_gain": rywak_fuse_odom_v_gain,
            "fuse_odom_w_gain": rywak_fuse_odom_w_gain,
            "vel_ema_alpha": rywak_vel_ema_alpha,
            "anchor_yaw_to_odom": rywak_anchor_yaw_to_odom,
            "anchor_yaw_to_odom_gain": rywak_anchor_yaw_to_odom_gain,
            "anchor_xy_to_odom": rywak_anchor_xy_to_odom,
            "anchor_xy_to_odom_gain": rywak_anchor_xy_to_odom_gain,
            "heading_for_xy_odom_weight": rywak_heading_for_xy_odom_weight,
            "xy_step_odom_weight": rywak_xy_step_odom_weight,
            "xy_step_odom_gain": rywak_xy_step_odom_gain,
            "max_integration_dt": rywak_max_integration_dt,
            "max_step_trans": rywak_max_step_trans,
            "max_step_yaw": rywak_max_step_yaw,
            "odom_rebase_to_local_origin": rywak_odom_rebase_to_local_origin,
            "use_odom_corrections": rywak_use_odom_corrections,
            "force_odom_pose": rywak_force_odom_pose,
            "odom_fallback_before_model_ready": rywak_odom_fallback_before_model_ready,
            "odom_guard_enabled": rywak_odom_guard_enabled,
            "odom_guard_fuse_weight": rywak_odom_guard_fuse_weight,
            "odom_guard_v_abs_diff": rywak_odom_guard_v_abs_diff,
            "odom_guard_v_rel_diff": rywak_odom_guard_v_rel_diff,
            "odom_guard_w_abs_diff": rywak_odom_guard_w_abs_diff,
            "odom_guard_w_rel_diff": rywak_odom_guard_w_rel_diff,
            "odom_guard_sign_conflict_speed": rywak_odom_guard_sign_conflict_speed,
            "odom_guard_xy_error_m": rywak_odom_guard_xy_error_m,
            "odom_guard_xy_anchor_base": rywak_odom_guard_xy_anchor_base,
            "odom_guard_xy_anchor_gain": rywak_odom_guard_xy_anchor_gain,
            "odom_guard_yaw_error_rad": rywak_odom_guard_yaw_error_rad,
            "odom_guard_yaw_anchor_base": rywak_odom_guard_yaw_anchor_base,
            "odom_guard_yaw_anchor_gain": rywak_odom_guard_yaw_anchor_gain,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(rywak_test_enabled).lower()),
    )
    infer_rywak_no_slam = Node(
        package="ai_slam_ai",
        executable="infer_rywak_node",
        parameters=[{
            "use_sim_time": True,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "model_source_experiment_id": model_source_experiment_id,
            "model_name": rywak_model_name,
            "scan_topic": "/scan_slam_rywak",
            "pose_topic": rywak_pose_topic_no_slam,
            "tf_parent": rywak_tf_parent_no_slam,
            "tf_child": rywak_tf_child_no_slam,
            "odom_topic": rywak_no_slam_infer_odom_topic,
            "init_from_odom_topic": rywak_no_slam_infer_init_from_odom_topic,
            "sync_tolerance_sec": rywak_no_slam_sync_tolerance,
            "interpolate_odom": rywak_no_slam_interpolate_odom,
            "sync_pair_gap_sec": rywak_no_slam_sync_pair_gap,
            "delta_scan_clip": rywak_no_slam_delta_scan_clip,
            "v_clip_abs": rywak_no_slam_v_clip_abs,
            "w_clip_abs": rywak_no_slam_w_clip_abs,
            "fuse_odom_v_weight": rywak_no_slam_fuse_odom_v_weight,
            "fuse_odom_w_weight": rywak_no_slam_fuse_odom_w_weight,
            "fuse_odom_v_gain": rywak_no_slam_fuse_odom_v_gain,
            "fuse_odom_w_gain": rywak_no_slam_fuse_odom_w_gain,
            "vel_ema_alpha": rywak_no_slam_vel_ema_alpha,
            "anchor_yaw_to_odom": rywak_no_slam_anchor_yaw_to_odom,
            "anchor_yaw_to_odom_gain": rywak_no_slam_anchor_yaw_to_odom_gain,
            "anchor_xy_to_odom": rywak_no_slam_anchor_xy_to_odom,
            "anchor_xy_to_odom_gain": rywak_no_slam_anchor_xy_to_odom_gain,
            "heading_for_xy_odom_weight": rywak_no_slam_heading_for_xy_odom_weight,
            "xy_step_odom_weight": rywak_no_slam_xy_step_odom_weight,
            "xy_step_odom_gain": rywak_no_slam_xy_step_odom_gain,
            "max_integration_dt": rywak_no_slam_max_integration_dt,
            "max_step_trans": rywak_no_slam_max_step_trans,
            "max_step_yaw": rywak_no_slam_max_step_yaw,
            "odom_rebase_to_local_origin": rywak_no_slam_odom_rebase_to_local_origin,
            "use_odom_corrections": rywak_no_slam_use_odom_corrections,
            "force_odom_pose": rywak_no_slam_force_odom_pose,
            "odom_fallback_before_model_ready": rywak_no_slam_odom_fallback_before_model_ready,
            "odom_guard_enabled": rywak_no_slam_odom_guard_enabled,
            "odom_guard_fuse_weight": rywak_no_slam_odom_guard_fuse_weight,
            "odom_guard_v_abs_diff": rywak_no_slam_odom_guard_v_abs_diff,
            "odom_guard_v_rel_diff": rywak_no_slam_odom_guard_v_rel_diff,
            "odom_guard_w_abs_diff": rywak_no_slam_odom_guard_w_abs_diff,
            "odom_guard_w_rel_diff": rywak_no_slam_odom_guard_w_rel_diff,
            "odom_guard_sign_conflict_speed": rywak_no_slam_odom_guard_sign_conflict_speed,
            "odom_guard_xy_error_m": rywak_no_slam_odom_guard_xy_error_m,
            "odom_guard_xy_anchor_base": rywak_no_slam_odom_guard_xy_anchor_base,
            "odom_guard_xy_anchor_gain": rywak_no_slam_odom_guard_xy_anchor_gain,
            "odom_guard_yaw_error_rad": rywak_no_slam_odom_guard_yaw_error_rad,
            "odom_guard_yaw_anchor_base": rywak_no_slam_odom_guard_yaw_anchor_base,
            "odom_guard_yaw_anchor_gain": rywak_no_slam_odom_guard_yaw_anchor_gain,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(rywak_no_slam_test_enabled).lower()),
    )
    evaluator = Node(
        package="ai_slam_eval",
        executable="eval_node",
        parameters=[{
            "use_sim_time": True,
            "seed": seed,
            "mode": mode,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "duration_sec": eval_duration_sec,
            "config_snapshot_path": resolved_config_path,
            "reference_map_yaml": reference_map_yaml,
            "spawn_x": spawn_pose["x"],
            "spawn_y": spawn_pose["y"],
            "spawn_yaw": spawn_pose["yaw"],
            "gt_world_frame_hint": gt_world_frame_hint,
            "world_name": world_name,
            "evaluation_label": evaluation_label,
            "artifact_subdir": evaluation_output_subdir,
            "finalize_experiment": finalize_experiment,
            "write_experiment_metadata": write_evaluation_metadata,
            "sync_tolerance_sec": eval_sync_tolerance,
            "maps_rotate_180": eval_maps_rotate_180,
            "maps_max_cols": eval_maps_max_cols,
            "warmup_sec": eval_warmup_sec,
            "points_min_translation": eval_points_min_translation,
            "points_min_rotation": eval_points_min_rotation,
            "points_min_time_gap_sec": eval_points_min_time_gap_sec,
            "points_filter_mode": eval_points_filter_mode,
            "points_use_probabilities": eval_points_use_probabilities,
            "points_occ_logodds_hit": eval_points_occ_logodds_hit,
            "points_free_logodds_miss": eval_points_free_logodds_miss,
            "points_logodds_min": eval_points_logodds_min,
            "points_logodds_max": eval_points_logodds_max,
            "gt_jump_filter_enabled": eval_gt_jump_filter_enabled,
            "gt_jump_filter_max_step_m": eval_gt_jump_filter_max_step_m,
            "pose_topic_ai": infer_pose_topic,
            "pose_topic_scanmatch": "/pose_scanmatch",
            "pose_topic_bruteforce": "/pose_bruteforce",
            "pose_topic_robak": robak_pose_topic,
            "pose_topic_rywak": rywak_pose_topic,
            "pose_topic_robak_no_slam": robak_pose_topic_no_slam,
            "pose_topic_rywak_no_slam": rywak_pose_topic_no_slam,
            "slam_baseline_tf_topic": "/tf",
            "slam_baseline_map_frame": slam_baseline_map_frame,
            "slam_baseline_odom_frame": slam_baseline_odom_frame,
            "slam_robak_map_frame": slam_robak_map_frame,
            "slam_robak_odom_frame": slam_robak_odom_frame,
            "slam_rywak_map_frame": slam_rywak_map_frame,
            "slam_rywak_odom_frame": slam_rywak_odom_frame,
            "robak_dataset_name": robak_train_dataset_name,
            "robak_model_name": robak_model_name,
            "robak_history_name": robak_history_name,
            "rywak_dataset_name": rywak_train_dataset_name,
            "rywak_model_name": rywak_model_name,
            "rywak_history_name": rywak_history_name,
        }],
        output="screen",
        condition=IfCondition(str(do_eval_phase).lower()),

    )

    # Environment variables
    env_vars = [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.pathsep.join(gz_resource_paths)),
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", os.pathsep.join(gazebo_model_paths)),
        SetEnvironmentVariable("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json"),
        SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "4.5"),
        SetEnvironmentVariable("MESA_GLSL_VERSION_OVERRIDE", "450"),
        SetEnvironmentVariable("GLOG_minloglevel", "2"),
        SetEnvironmentVariable("GLOG_logtostderr", "1"),
        SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] [{name}]: {message}"),
    ]
    # =========================
    # AUTO SHUTDOWN (TRAIN) - czekaj aż WSZYSTKIE trenery zakończą
    # =========================
    train_targets = []
    if ai_train_enabled:
        train_targets.append(trainer)
    if robak_train_enabled:
        train_targets.append(trainer_robak)
    if rywak_train_enabled:
        train_targets.append(trainer_rywak)

    auto_shutdown_train_handlers = []
    _shutdown_state = {"requested": False}

    def _request_launch_shutdown(msg: str):
        if _shutdown_state["requested"]:
            return [LogInfo(msg=f"{msg} (shutdown already requested)")]
        _shutdown_state["requested"] = True
        return [
            LogInfo(msg=msg),
            # Avoid ROS adapter race by stopping Gazebo process first and letting launch
            # complete shutdown from required process exit path.
            ExecuteProcess(
                cmd=[
                    "/bin/bash",
                    "-lc",
                    (
                        "pkill -TERM -x gz || true; "
                        "pkill -TERM -f 'ruby .*/gz sim' || true; "
                        "sleep 0.5; "
                        "pkill -KILL -x gz || true; "
                        "pkill -KILL -f 'ruby .*/gz sim' || true"
                    ),
                ],
                output="screen",
            ),
        ]
    dataset_targets = []
    if ai_dataset_enabled:
        dataset_targets.append(dataset_rec)
    if robak_dataset_enabled:
        dataset_targets.append(dataset_rec_robak)
    if rywak_dataset_enabled:
        dataset_targets.append(dataset_rec_rywak)

    _dataset_state = {"done": 0, "total": len(dataset_targets)}

    if phase == "train":
        _train_phase_state = {
            "train_done": 0,
            "train_total": len(train_targets),
            "dataset_done": 0,
            "dataset_total": len(dataset_targets),
        }

        def _finish_train_phase_if_ready():
            train_done = _train_phase_state["train_done"]
            train_total = _train_phase_state["train_total"]
            dataset_done = _train_phase_state["dataset_done"]
            dataset_total = _train_phase_state["dataset_total"]
            if train_done >= train_total and dataset_done >= dataset_total:
                return _request_launch_shutdown(
                    f"[AUTO] Training phase complete "
                    f"(trainings {train_done}/{train_total}, datasets {dataset_done}/{dataset_total}). "
                    "Shutting down simulation..."
                )
            return [
                LogInfo(
                    msg=(
                        f"[AUTO] Training phase progress: "
                        f"trainings {train_done}/{train_total}, datasets {dataset_done}/{dataset_total}. "
                        "Waiting for remaining processes..."
                    )
                )
            ]

        def _on_train_target_exit(context, *args, **kwargs):
            _train_phase_state["train_done"] += 1
            return _finish_train_phase_if_ready()

        def _on_train_dataset_exit(context, *args, **kwargs):
            _train_phase_state["dataset_done"] += 1
            return _finish_train_phase_if_ready()

        for target in train_targets:
            auto_shutdown_train_handlers.append(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=target,
                        on_exit=[OpaqueFunction(function=_on_train_target_exit)]
                    ),
                    condition=IfCondition(str(do_train_phase).lower()),
                )
            )
        for target in dataset_targets:
            auto_shutdown_train_handlers.append(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=target,
                        on_exit=[OpaqueFunction(function=_on_train_dataset_exit)]
                    ),
                    condition=IfCondition(str(do_train_phase).lower()),
                )
            )
    elif len(train_targets) > 0 and not do_test_phase:
        _train_state = {"done": 0, "total": len(train_targets)}

        def _on_trainer_exit(context, *args, **kwargs):
            _train_state["done"] += 1
            done = _train_state["done"]
            total = _train_state["total"]

            if done >= total:
                return _request_launch_shutdown(
                    f"[AUTO] All trainings finished ({done}/{total}). Shutting down simulation..."
                )
            return [
                LogInfo(msg=f"[AUTO] Training process exited ({done}/{total}). Waiting for others...")
            ]

        for t in train_targets:
            auto_shutdown_train_handlers.append(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=t,
                        on_exit=[OpaqueFunction(function=_on_trainer_exit)]
                    ),
                    condition=IfCondition(str(do_train_phase).lower()),
                )
            )

    def _on_dataset_exit(context, *args, **kwargs):
        _dataset_state["done"] += 1
        done = _dataset_state["done"]
        total = _dataset_state["total"]

        if done >= total:
            return _request_launch_shutdown(
                f"[AUTO] Wszystkie datasety zapisane ({done}/{total}). Zamykanie symulacji..."
            )
        return [
            LogInfo(msg=f"[AUTO] Dataset recorder zakończył pracę ({done}/{total}). Czekam na pozostałe...")
        ]

    auto_shutdown_dataset_handlers = []
    if phase == "dataset" and _dataset_state["total"] > 0:
        for target in dataset_targets:
            auto_shutdown_dataset_handlers.append(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=target,
                        on_exit=[OpaqueFunction(function=_on_dataset_exit)]
                    ),
                    condition=IfCondition(str(phase == "dataset").lower()),
                )
            )
    if phase == "dataset" and dataset_motion_watchdog_enabled:
        def _on_dataset_watchdog_exit(context, *args, **kwargs):
            event = kwargs.get("event", None)
            if event is None and args:
                event = args[0]

            def _as_int_or_none(val):
                if val is None:
                    return None
                try:
                    return int(val)
                except Exception:
                    return None

            rc = None
            candidates = []
            for k in ("returncode", "exit_code", "return_code", "rc"):
                if k in kwargs:
                    candidates.append(kwargs.get(k))
            if event is not None:
                for attr in ("returncode", "exit_code", "return_code"):
                    if hasattr(event, attr):
                        candidates.append(getattr(event, attr))

            for cand in candidates:
                rc_int = _as_int_or_none(cand)
                if rc_int is not None:
                    rc = rc_int
                    break

            # If shutdown is already in progress (e.g. normal end or global SIGINT),
            # watchdog exit must not be treated as the trigger.
            if _shutdown_state["requested"]:
                return [LogInfo(msg=f"[AUTO] Dataset motion watchdog exited (rc={rc}); shutdown already in progress.")]

            # Watchdog should stay alive during dataset run. Any non-zero code or unreadable
            # code here is treated as fail-fast trigger for next round.
            if rc in (42, None) or (isinstance(rc, int) and rc != 0):
                reason = "brak ruchu robota" if rc == 42 else "watchdog exit"
                return _request_launch_shutdown(
                    f"[AUTO] Dataset motion watchdog przerwał przebieg ({reason}, rc={rc}). "
                    "Zamykanie symulacji..."
                )
            return [LogInfo(msg=f"[AUTO] Dataset motion watchdog exited cleanly (rc={rc}); pomijam.")]

        auto_shutdown_dataset_handlers.append(
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=dataset_motion_watchdog,
                    on_exit=[OpaqueFunction(function=_on_dataset_watchdog_exit)],
                ),
                condition=IfCondition(str(phase == "dataset").lower()),
            )
        )

    def _on_eval_exit(context, *args, **kwargs):
        return _request_launch_shutdown("[AUTO] Ewaluacja zakończona. Zamykanie symulacji...")

    auto_shutdown_eval = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=evaluator,
            on_exit=[OpaqueFunction(function=_on_eval_exit)]
        ),
        condition=IfCondition(str(do_eval_phase).lower())
    )
    sim_stack_actions = [
        gz_launch_headless,
        gz_launch_gui,
        TimerAction(period=bridge_delay, actions=[bridge, bridge_tf_world]),
        robot_state_pub,
        # 4x scan_fix (baseline + osobne tory)
        scan_fix_baseline,
        scan_fix_ai,
        scan_fix_robak,
        scan_fix_rywak,
        # scan-matcher tory
        scan_matcher_local,
        scan_matcher_bruteforce,
        gt_pose,
        odom_corruptor,
        TimerAction(period=driver_start_delay, actions=[driver_auto, driver_planned]),
        # SLAM toolbox
        slam_baseline,
        slam_ai,
        slam_robak,
        slam_rywak,
        # lifecycle transitions
        configure_baseline,
        activate_baseline,
        configure_ai,
        activate_ai,
        configure_robak,
        activate_robak,
        configure_rywak,
        activate_rywak,
        # pipeline nodes that depend on simulator
        dataset_motion_watchdog,
        dataset_rec,
        infer,
        dataset_rec_robak,
        infer_robak,
        infer_robak_no_slam,
        dataset_rec_rywak,
        infer_rywak,
        infer_rywak_no_slam,
        evaluator,
    ]

    return [
        *env_vars,
        GroupAction(
            actions=sim_stack_actions,
            condition=IfCondition(str(not skip_simulation_for_external_train).lower()),
        ),
        trainer,
        trainer_robak,
        trainer_rywak,
        *auto_shutdown_dataset_handlers,
        *auto_shutdown_train_handlers,
        auto_shutdown_eval,
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value="experiment_config.yaml",
            description="Config file name (in config/) or full path"
        ),
        DeclareLaunchArgument("mode", default_value="__USE_CONFIG__", description="baseline|ai"),
        DeclareLaunchArgument("phase", default_value="__USE_CONFIG__", description="full|train|test|dataset"),
        DeclareLaunchArgument("world_sdf", default_value="__USE_CONFIG__", description="Gazebo world SDF (filename in worlds/ or absolute path)"),
        DeclareLaunchArgument("reference_map_yaml", default_value="__USE_CONFIG__", description="Reference map YAML (filename in maps/ or absolute path)"),
        DeclareLaunchArgument("evaluation_label", default_value="__USE_CONFIG__", description="Human-readable evaluation scenario label"),
        DeclareLaunchArgument("evaluation_output_subdir", default_value="__USE_CONFIG__", description="Optional evaluation artifact subdirectory inside experiment output"),
        DeclareLaunchArgument("finalize_experiment", default_value="__USE_CONFIG__", description="Whether eval node should finalize experiment metadata and append summary"),
        DeclareLaunchArgument("write_evaluation_metadata", default_value="__USE_CONFIG__", description="Whether eval node should update experiment metadata"),
        DeclareLaunchArgument("seed", default_value="__USE_CONFIG__", description="Random seed"),
        DeclareLaunchArgument("eval_duration_sec", default_value="__USE_CONFIG__", description="Evaluation duration"),
        DeclareLaunchArgument("dataset_duration_sec", default_value="__USE_CONFIG__", description="Dataset duration"),
        DeclareLaunchArgument("gui", default_value="__USE_CONFIG__", description="Enable Gazebo GUI"),
        DeclareLaunchArgument("out_dir", default_value="__USE_CONFIG__", description="Output directory"),
        DeclareLaunchArgument("experiment_id", default_value="__USE_CONFIG__", description="Experiment ID"),
        DeclareLaunchArgument(
            "model_source_experiment_id",
            default_value="__USE_CONFIG__",
            description="Load *.pt from out/<this_id>/ (empty = same as experiment_id)",
        ),
        DeclareLaunchArgument(
            "dataset_source_experiment_id",
            default_value="__USE_CONFIG__",
            description="Train from out/<this_id>/dataset*.npz (empty = collect local dataset)",
        ),
        OpaqueFunction(function=launch_setup),
    ])
