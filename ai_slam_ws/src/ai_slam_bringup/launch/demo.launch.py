"""
AI SLAM Demo Launch File - z centralną konfiguracją

Użycie:
  # Domyślna konfiguracja:
  ros2 launch ai_slam_bringup demo.launch.py
  
  # Własna konfiguracja:
  ros2 launch ai_slam_bringup demo.launch.py config:=experiment_config.yaml
  ros2 launch ai_slam_bringup demo.launch.py config:=fast_test.yaml
  
  # Override pojedynczych parametrów:
  ros2 launch ai_slam_bringup demo.launch.py mode:=baseline duration_sec:=60
  ros2 launch ai_slam_bringup demo.launch.py config:=fast_test.yaml seed:=999
"""
import os
import yaml
import math
import re
from datetime import datetime
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, TimerAction, SetEnvironmentVariable, 
    IncludeLaunchDescription, EmitEvent, LogInfo, RegisterEventHandler,
    OpaqueFunction
)
from launch.conditions import IfCondition
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


def load_config(config_file: str) -> dict:
    """Wczytuje konfigurację z pliku YAML."""
    bringup_share = get_package_share_directory("ai_slam_bringup")
    
    # Jeśli podano tylko nazwę pliku, szukaj w config/
    if not os.path.isabs(config_file):
        config_file = os.path.join(bringup_share, "config", config_file)
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


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


def merge_params(*param_dicts):
    """Łączy słowniki parametrów, ignorując wartości niebędące dict."""
    merged = {}
    for params in param_dicts:
        if isinstance(params, dict):
            merged.update(params)
    return merged


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
    train_world_sdf = str(get_config_value(cfg, "simulation", "train_world", default="world_train_house.sdf"))
    test_world_sdf = str(get_config_value(cfg, "simulation", "test_world", default="world_test_house.sdf"))

    # === CZASY ===
    eval_duration_sec = float(get_param("eval_duration_sec", ["timing", "eval_duration"], 60.0))
    dataset_duration_sec = float(get_param("dataset_duration_sec", ["timing", "dataset_duration"], 45.0))
    dataset_wait_timeout = float(get_config_value(cfg, "timing", "dataset_wait_timeout", default=120.0))
    bridge_delay = float(get_config_value(cfg, "timing", "bridge_delay", default=3.0))
    spawn_delay = float(get_config_value(cfg, "timing", "spawn_delay", default=5.0))
    slam_configure_delay = float(get_config_value(cfg, "timing", "slam_configure_delay", default=2.0))
    
    # === TRENING ===
    max_epochs = int(get_config_value(cfg, "training", "max_epochs", default=200))
    patience = int(get_config_value(cfg, "training", "patience", default=20))
    min_delta = float(get_config_value(cfg, "training", "min_delta", default=1e-5))
    learning_rate = float(get_config_value(cfg, "training", "learning_rate", default=0.001))
    batch_size = int(get_config_value(cfg, "training", "batch_size", default=128))
    validation_ratio = float(get_config_value(cfg, "training", "validation_ratio", default=0.2))
    skip_if_model_exists = parse_bool(
        get_config_value(cfg, "training", "skip_if_model_exists", default=True),
        default=True,
    )
    
    # === DATASET ===
    dataset_max_samples = int(get_config_value(cfg, "dataset", "max_samples", default=5000))
    dataset_scan_topic = str(get_config_value(cfg, "dataset", "scan_topic", default="/scan"))
    dataset_odom_topic = str(get_config_value(cfg, "dataset", "odom_topic", default="/odom"))
    dataset_gt_topic = str(get_config_value(cfg, "dataset", "gt_topic", default="/ground_truth_pose"))
    gt_cfg = get_config_value(cfg, "ground_truth", default={})
    gt_use_tf_world = parse_bool(gt_cfg.get("use_tf_world", True), default=True)
    gt_tf_world_topic = str(gt_cfg.get("tf_world_topic", "/tf_world"))
    gt_tf_world_timeout = float(gt_cfg.get("tf_world_timeout_sec", 0.5))
    gt_model_name_hint = str(gt_cfg.get("model_name_hint", "diffbot"))
    gt_base_link_hint = str(gt_cfg.get("base_link_hint", "base_link"))
    gt_world_frame_hint = str(gt_cfg.get("world_frame_hint", "world"))
    gt_heuristic_max_score = float(gt_cfg.get("heuristic_max_score", 12.0))
    gt_heuristic_max_step = float(gt_cfg.get("heuristic_max_step_m", 0.8))
    gt_debug_every_n = int(gt_cfg.get("debug_every_n", 2000))
    
    # === INFERENCE ===
    model_wait_timeout = float(get_config_value(cfg, "inference", "model_wait_timeout", default=300.0))
    infer_scan_topic = get_config_value(cfg, "inference", "scan_topic", default="/scan_slam_ai")
    infer_odom_topic = get_config_value(cfg, "inference", "odom_topic", default="/odom")

    # UWAGA: mapujemy nazwy z YAML -> nazwy parametrów w infer_node.py
    infer_pose_topic = get_config_value(cfg, "inference", "output_pose_topic", default="/pose_ai")
    infer_odom_ai_topic = get_config_value(cfg, "inference", "output_odom_topic", default="/odom_ai")
    infer_tf_parent = get_config_value(cfg, "inference", "tf_parent_frame", default="odom_ai")
    infer_tf_child  = get_config_value(cfg, "inference", "tf_child_frame", default="base_link_ai")

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
    driver_debug = parse_bool(get_config_value(cfg, "driver", "debug", default=True), default=True)
    driver_debug_every_n = int(get_config_value(cfg, "driver", "debug_every_n", default=10))

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
    robak_max_pair_dist = float(robak_cfg.get("max_pair_dist", robak_cfg.get("max_delta_dist", 0.5)))
    robak_max_pair_dyaw = float(robak_cfg.get("max_pair_dyaw", robak_cfg.get("max_delta_yaw", math.pi)))
    robak_label_frame = str(robak_cfg.get("label_frame", "local"))
    robak_sync_tolerance = float(robak_cfg.get("sync_tolerance_sec", 0.08))
    robak_aug_noise_std_scale = float(robak_cfg.get("augment_noise_std_scale", 0.0))
    robak_aug_cut_fraction = float(robak_cfg.get("augment_cut_fraction", 0.0))
    robak_aug_cut_max_points = int(robak_cfg.get("augment_cut_max_points", 20))
    robak_infer_max_step_trans = float(robak_cfg.get("infer_max_step_trans", 0.12))
    robak_infer_max_step_yaw = float(robak_cfg.get("infer_max_step_yaw", 0.35))
    robak_infer_delta_ema_alpha = float(robak_cfg.get("infer_delta_ema_alpha", 0.55))
    robak_infer_odom_heading_alpha = float(robak_cfg.get("infer_odom_heading_alpha", 0.20))
    robak_infer_odom_sync_tolerance = float(robak_cfg.get("infer_odom_sync_tolerance_sec", 0.08))
    robak_infer_odom_delta_xy_alpha = float(robak_cfg.get("infer_odom_delta_xy_alpha", 0.35))
    robak_infer_odom_delta_yaw_alpha = float(robak_cfg.get("infer_odom_delta_yaw_alpha", 0.45))
    robak_infer_odom_pose_xy_alpha = float(robak_cfg.get("infer_odom_pose_xy_alpha", 0.0))
    robak_infer_odom_pose_xy_gain = float(robak_cfg.get("infer_odom_pose_xy_gain", 0.0))
    robak_lr = float(robak_cfg.get("lr", learning_rate))
    robak_epochs = int(robak_cfg.get("max_epochs", max_epochs))
    robak_patience = int(robak_cfg.get("patience", patience))
    robak_val_ratio = float(robak_cfg.get("val_ratio", validation_ratio))
    robak_batch = int(robak_cfg.get("batch_size", batch_size))
    robak_pose_topic = str(robak_cfg.get("pose_topic", "/pose_robak"))

    # === RYWAK ===
    rywak_cfg = get_config_value(cfg, "rywak", default={})
    rywak_dataset_name = str(rywak_cfg.get("dataset_name", "dataset_rywak.npz"))
    rywak_model_name = str(rywak_cfg.get("model_name", "model_rywak.pt"))
    rywak_history_name = str(rywak_cfg.get("history_name", "train_history_rywak.json"))
    rywak_dataset_duration = float(rywak_cfg.get("dataset_duration", dataset_duration_sec))
    rywak_max_samples = int(rywak_cfg.get("max_samples", dataset_max_samples))
    rywak_odom_label_topic = str(rywak_cfg.get("odom_label_topic", "/odom_raw"))
    rywak_sync_tolerance = float(rywak_cfg.get("sync_tolerance_sec", 0.08))
    rywak_interpolate_odom = parse_bool(rywak_cfg.get("interpolate_odom", False), default=False)
    rywak_sync_pair_gap = float(rywak_cfg.get("sync_pair_gap_sec", 0.2))
    rywak_delta_scan_clip = float(rywak_cfg.get("delta_scan_clip", 2.0))
    rywak_min_sample_dist = float(rywak_cfg.get("min_sample_dist", 0.0))
    rywak_min_sample_dyaw = float(rywak_cfg.get("min_sample_dyaw", 0.0))
    rywak_min_sample_dt_sec = float(rywak_cfg.get("min_sample_dt_sec", 0.0))
    rywak_min_delta_scan_rms = float(rywak_cfg.get("min_delta_scan_rms", 0.0))
    rywak_sample_filter_mode = str(rywak_cfg.get("sample_filter_mode", "any"))
    rywak_hidden_dims = list(rywak_cfg.get("hidden_dims", [192, 96, 48]))
    rywak_dropout = float(rywak_cfg.get("dropout", 0.1))
    rywak_weight_decay = float(rywak_cfg.get("weight_decay", 1e-4))
    rywak_huber_delta = float(rywak_cfg.get("huber_delta", 1.0))
    rywak_input_noise_std = float(rywak_cfg.get("input_noise_std", 0.02))
    rywak_clip_grad_norm = float(rywak_cfg.get("clip_grad_norm", 1.0))
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
    rywak_anchor_xy_to_odom = float(rywak_cfg.get("anchor_xy_to_odom", 0.0))
    rywak_anchor_xy_to_odom_gain = float(rywak_cfg.get("anchor_xy_to_odom_gain", 0.0))
    rywak_heading_for_xy_odom_weight = float(rywak_cfg.get("heading_for_xy_odom_weight", 0.60))
    rywak_xy_step_odom_weight = float(rywak_cfg.get("xy_step_odom_weight", 0.35))
    rywak_xy_step_odom_gain = float(rywak_cfg.get("xy_step_odom_gain", 0.45))
    rywak_max_integration_dt = float(rywak_cfg.get("max_integration_dt", 0.20))
    rywak_lr = float(rywak_cfg.get("lr", learning_rate))
    rywak_epochs = int(rywak_cfg.get("max_epochs", max_epochs))
    rywak_patience = int(rywak_cfg.get("patience", patience))
    rywak_val_ratio = float(rywak_cfg.get("val_ratio", validation_ratio))
    rywak_batch = int(rywak_cfg.get("batch_size", batch_size))
    rywak_pose_topic = str(rywak_cfg.get("pose_topic", "/pose_rywak"))

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

    slam_baseline_params = merge_params(slam_global_cfg, slam_common_cfg, slam_baseline_cfg)
    slam_ai_params = merge_params(slam_global_cfg, slam_common_cfg, slam_ai_cfg)
    slam_robak_params = merge_params(slam_global_cfg, slam_common_cfg, slam_robak_cfg)
    slam_rywak_params = merge_params(slam_global_cfg, slam_common_cfg, slam_rywak_cfg)
        
    # === OUTPUT ===
    out_dir = str(get_param("out_dir", ["output", "base_dir"], "out"))
    # experiment_id: jeśli nie podano w launch args, generuj automatycznie
    experiment_id_launch = LaunchConfiguration("experiment_id").perform(context)
    if experiment_id_launch and experiment_id_launch != "__USE_CONFIG__":
        experiment_id = experiment_id_launch
    else:
        experiment_id = generate_experiment_id()
    
    # === ŚCIEŻKI ===
    gazebo_share = get_package_share_directory("ai_slam_gazebo")
    desc_share = get_package_share_directory("ai_slam_description")
    bringup_share = get_package_share_directory("ai_slam_bringup")
    eval_share = get_package_share_directory("ai_slam_eval")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    # World (można podać nazwę .sdf z ai_slam_gazebo/worlds/ lub ścieżkę absolutną)
    # World (launch-arg world_sdf ma pierwszeństwo nad configiem)
    world_path_cfg = str(get_config_value(cfg, "simulation", "world_path", default=""))

    if world_path_cfg:
        world_path = world_path_cfg if os.path.isabs(world_path_cfg) else os.path.join(gazebo_share, world_path_cfg)
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

        if os.path.isabs(selected_world_sdf):
            world_path = selected_world_sdf
        else:
            world_path = os.path.join(gazebo_share, "worlds", selected_world_sdf)
    world_name = extract_world_name(world_path)

    bridge_cfg = os.path.join(gazebo_share, "config", "bridge.yaml")
    model_sdf = os.path.join(desc_share, "models", "diffbot.sdf")
    urdf_path = os.path.join(desc_share, "urdf", "diffbot.urdf")
    slam_params_baseline = os.path.join(bringup_share, "config", "slam_toolbox_baseline.yaml")
    slam_params_ai = os.path.join(bringup_share, "config", "slam_toolbox_ai.yaml")
    
    # Reference map
    ref_map_cfg = str(get_config_value(cfg, "evaluation", "reference_map_yaml", default=""))
    reference_map_yaml = ref_map_cfg if ref_map_cfg else os.path.join(eval_share, "maps", "reference_map.yaml")
    eval_sync_tolerance = float(get_config_value(cfg, "evaluation", "sync_tolerance_sec", default=0.15))
    eval_maps_rotate_180 = parse_bool(get_config_value(cfg, "evaluation", "maps_rotate_180", default=True), default=True)
    eval_maps_max_cols = int(get_config_value(cfg, "evaluation", "maps_max_cols", default=3))
    eval_points_min_translation = float(get_config_value(cfg, "evaluation", "points_min_translation", default=0.0))
    eval_points_min_rotation = float(get_config_value(cfg, "evaluation", "points_min_rotation", default=0.0))
    eval_points_min_time_gap_sec = float(get_config_value(cfg, "evaluation", "points_min_time_gap_sec", default=0.0))
    eval_points_filter_mode = str(get_config_value(cfg, "evaluation", "points_filter_mode", default="any"))
    
    gz_sim_launch_py = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    # === LOG KONFIGURACJI ===
    print("\n" + "="*70)
    print("AI SLAM EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"  Config file: {config_file or 'none (defaults)'}")
    print(f"  Mode: {mode}")
    print(f"  Seed: {seed}")
    print(f"  GUI: {gui}")
    print(f"  Eval duration: {eval_duration_sec}s")
    print(f"  Dataset duration: {dataset_duration_sec}s")
    print(f"  Training: max_epochs={max_epochs}, patience={patience}, lr={learning_rate}")
    print(f"  Output: {out_dir}/{experiment_id}")
    print("="*70 + "\n")

    # === URDF ===
    with open(urdf_path, "r", encoding="utf-8") as f:
        robot_description = f.read()

    # === NODES ===
    is_ai_mode = (mode == "ai")
    is_gui = (gui == "true")
    tracks_cfg = get_config_value(cfg, "tracks", default={})

    tor3_local_enabled = bool(tracks_cfg.get("tor3_local", True))
    tor4_bruteforce_enabled = bool(tracks_cfg.get("tor4_bruteforce", False))
    tor5_robak_enabled = bool(tracks_cfg.get("tor5_robak", False))
    tor6_rywak_enabled = bool(tracks_cfg.get("tor6_rywak", False))

    # fazy (zakładam, że zmienną `phase` już wcześniej wyliczysz z get_param)
    do_dataset_phase = is_ai_mode and (phase in ("full", "train", "dataset"))
    do_train_phase = is_ai_mode and (phase in ("full", "train"))
    do_test_phase  = is_ai_mode and (phase in ("full", "test"))
    do_eval_phase  = (phase in ("full", "test"))
    do_train_only = is_ai_mode and (phase == "train")
    # Robak / Rywak: osobne fazy train/test
    robak_dataset_enabled = tor5_robak_enabled and do_dataset_phase
    robak_train_enabled = tor5_robak_enabled and do_train_phase
    robak_test_enabled  = tor5_robak_enabled and do_test_phase
    rywak_dataset_enabled = tor6_rywak_enabled and do_dataset_phase
    rywak_train_enabled = tor6_rywak_enabled and do_train_phase
    rywak_test_enabled  = tor6_rywak_enabled and do_test_phase
    # tracki tylko w test/full (w train oszczędzamy CPU)
    tor3_local_enabled = tor3_local_enabled and do_eval_phase
    tor4_bruteforce_enabled = tor4_bruteforce_enabled and do_eval_phase
    # Gazebo launch
    gz_launch_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch_py),
        launch_arguments={
            "gz_args": f"{world_path} -r -s --headless-rendering",
            "on_exit_shutdown": "True",
        }.items(),
        condition=IfCondition(str(not is_gui).lower()),
    )
    
    gz_launch_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch_py),
        launch_arguments={
            "gz_args": f"{world_path} -r",
            "on_exit_shutdown": "True",
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

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", "diffbot", "-file", model_sdf, "-x", "0", "-y", "0", "-z", "0.10"],
        output="screen",
        shell=True,
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
        condition=IfCondition(str(do_test_phase).lower()),
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
        condition=IfCondition(str(robak_test_enabled).lower()),
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
        condition=IfCondition(str(rywak_test_enabled).lower()),
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
            "tf_world_timeout_sec": gt_tf_world_timeout,
            "model_name_hint": gt_model_name_hint,
            "base_link_hint": gt_base_link_hint,
            "world_frame_hint": gt_world_frame_hint,
            "heuristic_max_score": gt_heuristic_max_score,
            "heuristic_max_step_m": gt_heuristic_max_step,
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

    driver = Node(
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
            "debug": driver_debug,
            "debug_every_n": driver_debug_every_n,
            "odom_topic": odom_in_topic,  # zwykle /odom_raw
        }],
        output="screen",
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
        condition=IfCondition(str(do_test_phase).lower()),
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
        ]
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
        )
    )

    configure_ai = TimerAction(
        period=slam_configure_delay,
        actions=[
            EmitEvent(event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_ai),
                transition_id=Transition.TRANSITION_CONFIGURE
            ))
        ],
        condition=IfCondition(str(do_test_phase).lower()),
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
        condition=IfCondition(str(do_test_phase).lower()),
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
        }],
        output="screen",
        condition=IfCondition(str(do_dataset_phase).lower()),
    )

    trainer = Node(
        package="ai_slam_ai",
        executable="train_model",
        parameters=[{
            "use_sim_time": True, 
            "seed": seed, 
            "out_dir": out_dir, 
            "experiment_id": experiment_id,
            "skip_if_model_exists": skip_if_model_exists,
            "dataset_wait_timeout": dataset_wait_timeout,
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "lr": learning_rate,
            "batch_size": batch_size,
            "val_ratio": validation_ratio,
        }],
        output="screen",
        condition=IfCondition(str(do_train_phase).lower()),
    )

    infer = Node(
        package="ai_slam_ai",
        executable="infer_node",
        parameters=[{
            "use_sim_time": True, 
            "seed": seed, 
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "model_wait_timeout": model_wait_timeout,
            "scan_topic": infer_scan_topic,
            "odom_topic": infer_odom_topic,
            "pose_topic": infer_pose_topic,         # zgodnie z infer_node.py :contentReference[oaicite:18]{index=18}
            "odom_ai_topic": infer_odom_ai_topic,   # zgodnie z infer_node.py :contentReference[oaicite:19]{index=19}
            "tf_parent": infer_tf_parent,           # zgodnie z infer_node.py :contentReference[oaicite:20]{index=20}
            "tf_child": infer_tf_child,             # zgodnie z infer_node.py :contentReference[oaicite:21]{index=21}
   
        }],
        output="screen",
        condition=IfCondition(str(do_test_phase).lower()),
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
            "scan_topic": dataset_scan_topic,
            "gt_topic": dataset_gt_topic,
            "dataset_name": robak_dataset_name,
            "offsets": robak_offsets,
            "min_pair_dist": robak_min_pair_dist,
            "min_pair_dyaw": robak_min_pair_dyaw,
            "min_pair_dt_sec": robak_min_pair_dt_sec,
            "pair_filter_mode": robak_pair_filter_mode,
            "max_pair_dist": robak_max_pair_dist,
            "max_pair_dyaw": robak_max_pair_dyaw,
            "label_frame": robak_label_frame,
            "sync_tolerance_sec": robak_sync_tolerance,
            "augment_noise_std_scale": robak_aug_noise_std_scale,
            "augment_cut_fraction": robak_aug_cut_fraction,
            "augment_cut_max_points": robak_aug_cut_max_points,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(robak_dataset_enabled).lower()),
    )

    trainer_robak = Node(
        package="ai_slam_ai",
        executable="train_model_robak",
        parameters=[{
            "use_sim_time": True,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "dataset_name": robak_dataset_name,
            "model_name": robak_model_name,
            "history_name": robak_history_name,
            "skip_if_model_exists": skip_if_model_exists,
            "dataset_wait_timeout": dataset_wait_timeout,
            "max_epochs": robak_epochs,
            "patience": robak_patience,
            "min_delta": min_delta,
            "lr": robak_lr,
            "batch_size": robak_batch,
            "val_ratio": robak_val_ratio,
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
            "model_name": robak_model_name,
            "scan_topic": "/scan_slam_robak",
            "pose_topic": robak_pose_topic,
            "init_from": "gt",
            "gt_topic": dataset_gt_topic,
            "odom_topic": odom_in_topic,
            "max_step_trans": robak_infer_max_step_trans,
            "max_step_yaw": robak_infer_max_step_yaw,
            "delta_ema_alpha": robak_infer_delta_ema_alpha,
            "odom_heading_alpha": robak_infer_odom_heading_alpha,
            "odom_sync_tolerance_sec": robak_infer_odom_sync_tolerance,
            "odom_delta_xy_alpha": robak_infer_odom_delta_xy_alpha,
            "odom_delta_yaw_alpha": robak_infer_odom_delta_yaw_alpha,
            "odom_pose_xy_alpha": robak_infer_odom_pose_xy_alpha,
            "odom_pose_xy_gain": robak_infer_odom_pose_xy_gain,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(robak_test_enabled).lower()),
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
            "scan_topic": dataset_scan_topic,
            "odom_topic": rywak_odom_label_topic,
            "sync_tolerance_sec": rywak_sync_tolerance,
            "interpolate_odom": rywak_interpolate_odom,
            "sync_pair_gap_sec": rywak_sync_pair_gap,
            "delta_scan_clip": rywak_delta_scan_clip,
            "min_sample_dist": rywak_min_sample_dist,
            "min_sample_dyaw": rywak_min_sample_dyaw,
            "min_sample_dt_sec": rywak_min_sample_dt_sec,
            "min_delta_scan_rms": rywak_min_delta_scan_rms,
            "sample_filter_mode": rywak_sample_filter_mode,
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
            "use_sim_time": True,
            "seed": seed,
            "out_dir": out_dir,
            "experiment_id": experiment_id,
            "dataset_name": rywak_dataset_name,
            "model_name": rywak_model_name,
            "history_name": rywak_history_name,
            "skip_if_model_exists": skip_if_model_exists,
            "dataset_wait_timeout": dataset_wait_timeout,
            "max_epochs": rywak_epochs,
            "patience": rywak_patience,
            "min_delta": min_delta,
            "lr": rywak_lr,
            "batch_size": rywak_batch,
            "val_ratio": rywak_val_ratio,
            "hidden_dims": rywak_hidden_dims,
            "dropout": rywak_dropout,
            "weight_decay": rywak_weight_decay,
            "huber_delta": rywak_huber_delta,
            "input_noise_std": rywak_input_noise_std,
            "clip_grad_norm": rywak_clip_grad_norm,
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
            "model_name": rywak_model_name,
            "scan_topic": "/scan_slam_rywak",
            "pose_topic": rywak_pose_topic,
            "odom_topic": rywak_odom_label_topic,
            "init_from_odom_topic": odom_in_topic,
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
            "anchor_xy_to_odom": rywak_anchor_xy_to_odom,
            "anchor_xy_to_odom_gain": rywak_anchor_xy_to_odom_gain,
            "heading_for_xy_odom_weight": rywak_heading_for_xy_odom_weight,
            "xy_step_odom_weight": rywak_xy_step_odom_weight,
            "xy_step_odom_gain": rywak_xy_step_odom_gain,
            "max_integration_dt": rywak_max_integration_dt,
            "write_experiment_metadata": False,
        }],
        output="screen",
        condition=IfCondition(str(rywak_test_enabled).lower()),
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
            "sync_tolerance_sec": eval_sync_tolerance,
            "maps_rotate_180": eval_maps_rotate_180,
            "maps_max_cols": eval_maps_max_cols,
            "points_min_translation": eval_points_min_translation,
            "points_min_rotation": eval_points_min_rotation,
            "points_min_time_gap_sec": eval_points_min_time_gap_sec,
            "points_filter_mode": eval_points_filter_mode,
            "pose_topic_ai": infer_pose_topic,
            "pose_topic_scanmatch": "/pose_scanmatch",
            "pose_topic_bruteforce": "/pose_bruteforce",
            "pose_topic_robak": robak_pose_topic,
            "pose_topic_rywak": rywak_pose_topic,
            "robak_dataset_name": robak_dataset_name,
            "robak_model_name": robak_model_name,
            "robak_history_name": robak_history_name,
            "rywak_dataset_name": rywak_dataset_name,
            "rywak_model_name": rywak_model_name,
            "rywak_history_name": rywak_history_name,
        }],
        output="screen",
        condition=IfCondition(str(do_eval_phase).lower()),

    )

    # Environment variables
    env_vars = [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.pathsep.join([gazebo_share, desc_share])),
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
    if do_train_phase:
        train_targets.append(trainer)
    if robak_train_enabled:
        train_targets.append(trainer_robak)
    if rywak_train_enabled:
        train_targets.append(trainer_rywak)

    _train_state = {"done": 0, "total": len(train_targets)}

    def _on_trainer_exit(context, *args, **kwargs):
        _train_state["done"] += 1
        done = _train_state["done"]
        total = _train_state["total"]

        if done >= total:
            return [
                LogInfo(msg=f"[AUTO] All trainings finished ({done}/{total}). Shutting down simulation..."),
                EmitEvent(event=Shutdown())
            ]
        else:
            return [
                LogInfo(msg=f"[AUTO] Training process exited ({done}/{total}). Waiting for others...")
            ]

    auto_shutdown_train_handlers = []
    if _train_state["total"] > 0:
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

    dataset_targets = []
    if do_dataset_phase:
        dataset_targets.append(dataset_rec)
    if robak_dataset_enabled:
        dataset_targets.append(dataset_rec_robak)
    if rywak_dataset_enabled:
        dataset_targets.append(dataset_rec_rywak)

    _dataset_state = {"done": 0, "total": len(dataset_targets)}

    def _on_dataset_exit(context, *args, **kwargs):
        _dataset_state["done"] += 1
        done = _dataset_state["done"]
        total = _dataset_state["total"]

        if done >= total:
            return [
                LogInfo(msg=f"[AUTO] Wszystkie datasety zapisane ({done}/{total}). Zamykanie symulacji..."),
                EmitEvent(event=Shutdown())
            ]
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

    auto_shutdown_eval = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=evaluator,
            on_exit=[
                LogInfo(msg='[AUTO] Ewaluacja zakończona. Zamykanie symulacji...'),
                EmitEvent(event=Shutdown())
            ]
        ),
        condition=IfCondition(str(do_eval_phase).lower())
    )
    return [
        *env_vars,
        gz_launch_headless,
        gz_launch_gui,

        TimerAction(period=bridge_delay, actions=[bridge, bridge_tf_world]),
        TimerAction(period=spawn_delay, actions=[spawn]),

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
        driver,

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

        # pipeline dataset/train/infer
        dataset_rec,
        trainer,
        infer,

        dataset_rec_robak,
        trainer_robak,
        infer_robak,

        dataset_rec_rywak,
        trainer_rywak,
        infer_rywak,

        evaluator,

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
        DeclareLaunchArgument("seed", default_value="__USE_CONFIG__", description="Random seed"),
        DeclareLaunchArgument("eval_duration_sec", default_value="__USE_CONFIG__", description="Evaluation duration"),
        DeclareLaunchArgument("dataset_duration_sec", default_value="__USE_CONFIG__", description="Dataset duration"),
        DeclareLaunchArgument("gui", default_value="__USE_CONFIG__", description="Enable Gazebo GUI"),
        DeclareLaunchArgument("out_dir", default_value="__USE_CONFIG__", description="Output directory"),
        DeclareLaunchArgument("experiment_id", default_value="__USE_CONFIG__", description="Experiment ID"),
        OpaqueFunction(function=launch_setup),
    ])
