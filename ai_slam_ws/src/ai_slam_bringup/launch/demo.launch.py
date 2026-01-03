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
from datetime import datetime

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


def get_config_value(config: dict, *keys, default=None):
    """Bezpiecznie pobiera wartość z zagnieżdżonego słownika."""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def launch_setup(context, *args, **kwargs):
    """Funkcja setup wywoływana w runtime z dostępem do kontekstu."""
    
    # Pobierz wartości argumentów launch
    config_file = LaunchConfiguration("config").perform(context)
    
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
    
    # === CZASY ===
    duration_sec = float(get_param("duration_sec", ["timing", "experiment_duration"], 120.0))
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
    skip_if_model_exists = bool(get_config_value(cfg, "training", "skip_if_model_exists", default=True))
    
    # === DATASET ===
    dataset_max_samples = int(get_config_value(cfg, "dataset", "max_samples", default=5000))
    dataset_scan_topic = str(get_config_value(cfg, "dataset", "scan_topic", default="/scan"))
    dataset_odom_topic = str(get_config_value(cfg, "dataset", "odom_topic", default="/odom"))
    dataset_gt_topic = str(get_config_value(cfg, "dataset", "gt_topic", default="/ground_truth_pose"))
    
    # === INFERENCE ===
    model_wait_timeout = float(get_config_value(cfg, "inference", "model_wait_timeout", default=300.0))
    
    # === ODOMETRY ===
    rw_sigma_xy = float(get_config_value(cfg, "odometry", "rw_sigma_xy", default=0.005))
    rw_sigma_theta = float(get_config_value(cfg, "odometry", "rw_sigma_theta", default=0.003))
    
    # === DRIVER ===
    driver_linear_vel = float(get_config_value(cfg, "driver", "linear_velocity", default=0.3))
    driver_angular_vel = float(get_config_value(cfg, "driver", "angular_velocity", default=0.5))
    driver_turn_prob = float(get_config_value(cfg, "driver", "turn_probability", default=0.02))
    driver_obstacle_thresh = float(get_config_value(cfg, "driver", "obstacle_threshold", default=0.5))
    
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

    world_path = os.path.join(gazebo_share, "worlds", "ai_slam_world.sdf")
    bridge_cfg = os.path.join(gazebo_share, "config", "bridge.yaml")
    model_sdf = os.path.join(desc_share, "models", "diffbot.sdf")
    urdf_path = os.path.join(desc_share, "urdf", "diffbot.urdf")
    slam_params_baseline = os.path.join(bringup_share, "config", "slam_toolbox_baseline.yaml")
    slam_params_ai = os.path.join(bringup_share, "config", "slam_toolbox_ai.yaml")
    
    # Reference map
    ref_map_cfg = str(get_config_value(cfg, "evaluation", "reference_map_yaml", default=""))
    reference_map_yaml = ref_map_cfg if ref_map_cfg else os.path.join(eval_share, "maps", "reference_map.yaml")
    
    gz_sim_launch_py = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    # === LOG KONFIGURACJI ===
    print("\n" + "="*70)
    print("AI SLAM EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"  Config file: {config_file or 'none (defaults)'}")
    print(f"  Mode: {mode}")
    print(f"  Seed: {seed}")
    print(f"  GUI: {gui}")
    print(f"  Experiment duration: {duration_sec}s")
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

    scan_fix = Node(
        package="ai_slam_bringup",
        executable="scan_fix",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    gt_pose = Node(
        package="ai_slam_bringup",
        executable="gt_pose_publisher",
        parameters=[{"use_sim_time": True}],
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
        }],
        output="screen",
    )

    # SLAM Toolbox nodes - output="log" to suppress TF_OLD_DATA spam
    slam_baseline = LifecycleNode(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox_baseline",
        namespace="",
        parameters=[slam_params_baseline, {"use_sim_time": True}],
        output="log",  # Redirect to log file instead of screen
        arguments=["--ros-args", "--log-level", "warn"],
    )

    slam_ai = LifecycleNode(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox_ai",
        namespace="",
        parameters=[slam_params_ai, {"use_sim_time": True}],
        arguments=["--ros-args", "--log-level", "warn"],
        remappings=[("/map", "/map_ai")],
        output="log",  # Redirect to log file instead of screen
        condition=IfCondition(str(is_ai_mode).lower()),
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
        condition=IfCondition(str(is_ai_mode).lower()),
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
        condition=IfCondition(str(is_ai_mode).lower()),
    )

    lifecycle_mgr_baseline = Node(
        package="ai_slam_bringup",
        executable="lifecycle_manager",
        parameters=[{"use_sim_time": True, "nodes": ["slam_toolbox_baseline"]}],
        output="screen",
    )

    lifecycle_mgr_ai = Node(
        package="ai_slam_bringup",
        executable="lifecycle_manager",
        parameters=[{"use_sim_time": True, "nodes": ["slam_toolbox_ai"]}],
        output="screen",
        condition=IfCondition(str(is_ai_mode).lower()),
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
        condition=IfCondition(str(is_ai_mode).lower()),
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
        condition=IfCondition(str(is_ai_mode).lower()),
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
        }],
        output="screen",
        condition=IfCondition(str(is_ai_mode).lower()),
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
            "duration_sec": duration_sec,
            "reference_map_yaml": reference_map_yaml,
        }],
        output="screen",
    )

    # Environment variables
    env_vars = [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.pathsep.join([gazebo_share, desc_share])),
        SetEnvironmentVariable("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json"),
        SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "4.5"),
        SetEnvironmentVariable("MESA_GLSL_VERSION_OVERRIDE", "450"),
        SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] [{name}]: {message}"),
    ]

    return [
        *env_vars,
        gz_launch_headless,
        gz_launch_gui,
        TimerAction(period=bridge_delay, actions=[bridge]),
        TimerAction(period=spawn_delay, actions=[spawn]),
        robot_state_pub,
        scan_fix,
        gt_pose,
        odom_corruptor,
        driver,
        slam_baseline,
        slam_ai,
        configure_baseline,
        activate_baseline,
        configure_ai,
        activate_ai,
        lifecycle_mgr_baseline,
        lifecycle_mgr_ai,
        dataset_rec,
        trainer,
        infer,
        evaluator,
    ]


def generate_launch_description():
    return LaunchDescription([
        # Argument do podania pliku konfiguracyjnego
        DeclareLaunchArgument(
            "config", 
            default_value="experiment_config.yaml",
            description="Config file name (in config/) or full path"
        ),
        # Argumenty które mogą nadpisać config (puste = użyj z config)
        DeclareLaunchArgument("mode", default_value="__USE_CONFIG__", description="baseline|ai"),
        DeclareLaunchArgument("seed", default_value="__USE_CONFIG__", description="Random seed"),
        DeclareLaunchArgument("duration_sec", default_value="__USE_CONFIG__", description="Experiment duration"),
        DeclareLaunchArgument("dataset_duration_sec", default_value="__USE_CONFIG__", description="Dataset duration"),
        DeclareLaunchArgument("gui", default_value="__USE_CONFIG__", description="Enable Gazebo GUI"),
        DeclareLaunchArgument("out_dir", default_value="__USE_CONFIG__", description="Output directory"),
        DeclareLaunchArgument("experiment_id", default_value="__USE_CONFIG__", description="Experiment ID"),
        # OpaqueFunction pozwala na runtime evaluation
        OpaqueFunction(function=launch_setup),
    ])
