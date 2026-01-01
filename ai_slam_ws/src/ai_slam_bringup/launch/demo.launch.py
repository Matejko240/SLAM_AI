from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, SetEnvironmentVariable, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os



def generate_launch_description():
    mode = LaunchConfiguration("mode")
    seed = LaunchConfiguration("seed")
    duration_sec = LaunchConfiguration("duration_sec")
    dataset_duration_sec = LaunchConfiguration("dataset_duration_sec")

    gazebo_share = get_package_share_directory("ai_slam_gazebo")
    desc_share = get_package_share_directory("ai_slam_description")
    bringup_share = get_package_share_directory("ai_slam_bringup")
    eval_share = get_package_share_directory("ai_slam_eval")

    world_path = os.path.join(gazebo_share, "worlds", "ai_slam_world.sdf")
    bridge_cfg = os.path.join(gazebo_share, "config", "bridge.yaml")
    model_sdf = os.path.join(desc_share, "models", "diffbot.sdf")
    urdf_path = os.path.join(desc_share, "urdf", "diffbot.urdf")

    slam_params_baseline = os.path.join(bringup_share, "config", "slam_toolbox_baseline.yaml")
    slam_params_ai = os.path.join(bringup_share, "config", "slam_toolbox_ai.yaml")

    reference_map_yaml = os.path.join(eval_share, "maps", "reference_map.yaml")

    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    gz_sim_launch_py = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch_py),
        launch_arguments={
            "gz_args": [world_path, " -r"],
            "on_exit_shutdown": "True",
        }.items(),
    )


    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{"config_file": bridge_cfg}],
        output="screen",
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", "diffbot", "-file", model_sdf, "-x", "0", "-y", "0", "-z", "0.10"],
        output="screen",
    )

    with open(urdf_path, "r", encoding="utf-8") as f:
        robot_description = f.read()

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
        output="screen",
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
        parameters=[{"use_sim_time": True, "seed": seed}],
        output="screen",
    )

    driver = Node(
        package="ai_slam_bringup",
        executable="auto_driver",
        parameters=[{"use_sim_time": True, "seed": seed}],
        output="screen",
    )

    slam_baseline = Node(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox_baseline",
        parameters=[slam_params_baseline],
        output="screen",
    )

    slam_ai = Node(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox_ai",
        parameters=[slam_params_ai],
        remappings=[("/map", "/map_ai")],
        output="screen",
        condition=IfCondition(PythonExpression(['"', mode, '" == "ai"'])),
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
        condition=IfCondition(PythonExpression(['"', mode, '" == "ai"'])),
    )

    dataset_rec = Node(
        package="ai_slam_ai",
        executable="dataset_recorder",
        parameters=[
            {"use_sim_time": True, "seed": seed, "out_dir": "out", "duration_sec": dataset_duration_sec}
        ],
        output="screen",
        condition=IfCondition(PythonExpression(['"', mode, '" == "ai"'])),
    )

    trainer = Node(
        package="ai_slam_ai",
        executable="train_model",
        parameters=[{"use_sim_time": True, "seed": seed, "out_dir": "out", "skip_if_model_exists": True}],
        output="screen",
        condition=IfCondition(PythonExpression(['"', mode, '" == "ai"'])),
    )

    infer = Node(
        package="ai_slam_ai",
        executable="infer_node",
        parameters=[{"use_sim_time": True, "seed": seed, "out_dir": "out"}],
        output="screen",
        condition=IfCondition(PythonExpression(['"', mode, '" == "ai"'])),
    )

    evaluator = Node(
        package="ai_slam_eval",
        executable="eval_node",
        parameters=[
            {
                "use_sim_time": True,
                "seed": seed,
                "mode": mode,
                "out_dir": "out",
                "duration_sec": duration_sec,
                "reference_map_yaml": reference_map_yaml,
            }
        ],
        output="screen",
    )

    env = [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.pathsep.join([gazebo_share, desc_share])),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="baseline", description="baseline|ai"),
            DeclareLaunchArgument("seed", default_value="123", description="Deterministic seed"),
            DeclareLaunchArgument("duration_sec", default_value="120", description="Experiment duration"),
            DeclareLaunchArgument("dataset_duration_sec", default_value="45", description="Dataset recording duration (ai mode)"),
            *env,
            gz_launch,
            TimerAction(period=1.0, actions=[bridge]),
            TimerAction(period=2.0, actions=[spawn]),
            robot_state_pub,
            scan_fix,
            gt_pose,
            odom_corruptor,
            driver,
            slam_baseline,
            slam_ai,
            lifecycle_mgr_baseline,
            lifecycle_mgr_ai,
            dataset_rec,
            trainer,
            infer,
            evaluator,
        ]
    )
