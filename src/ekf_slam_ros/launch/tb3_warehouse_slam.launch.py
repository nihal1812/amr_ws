import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _join_paths(*paths):
    return os.pathsep.join([p for p in paths if p])


def generate_launch_description():
    # --- Paths ---
    gazebo_launch = os.path.join(
        get_package_share_directory("gazebo_ros"),
        "launch",
        "gazebo.launch.py",
    )

    # AWS warehouse world
    aws_root = get_package_share_directory("aws_robomaker_small_warehouse_world")
    world_default = os.path.join(
        aws_root,
        "worlds", "no_roof_small_warehouse", "no_roof_small_warehouse.world",
    )
    aws_models = os.path.join(aws_root, "models")

    # Turtlebot3 SDF model
    tb3_model_sdf_default = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"),
        "models", "turtlebot3_waffle_pi", "model.sdf",
    )

    # ekf_slam_ros params
    slam_params = os.path.join(
        get_package_share_directory("ekf_slam_ros"),
        "config", "slam_params.yaml",
    )

    # --- Launch args ---
    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")
    verbose = LaunchConfiguration("verbose")
    robot_name = LaunchConfiguration("robot_name")
    tb3_model_sdf = LaunchConfiguration("tb3_model_sdf")

    # Optional robot_state_publisher (URDF) if you still want RViz robot model
    use_rsp = LaunchConfiguration("use_rsp")

    # Map tool args
    map_mode = LaunchConfiguration("map_mode")  # none|save|load
    map_topic = LaunchConfiguration("map_topic")
    map_output_dir = LaunchConfiguration("map_output_dir")
    map_name = LaunchConfiguration("map_name")
    map_yaml_path = LaunchConfiguration("map_yaml_path")

    # --- Env: Gazebo model paths ---
    old_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    new_model_path = _join_paths(old_model_path, aws_models)

    use_sim_time_param = {"use_sim_time": True}

    # Conditions for map nodes
    do_map_save = IfCondition(PythonExpression(["'", map_mode, "' == 'save'"]))
    do_map_load = IfCondition(PythonExpression(["'", map_mode, "' == 'load'"]))
    do_rsp = IfCondition(use_rsp)

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=world_default),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("verbose", default_value="false"),
        DeclareLaunchArgument("robot_name", default_value="tb3_waffle_pi"),
        DeclareLaunchArgument("tb3_model_sdf", default_value=tb3_model_sdf_default),

        # Set to "true" only if you want robot_state_publisher (needs URDF available)
        DeclareLaunchArgument("use_rsp", default_value="false",
                              description="Publish TF/robot model via robot_state_publisher (requires URDF)"),

        # Map tool launch args
        DeclareLaunchArgument("map_mode", default_value="none",
                              description="none|save|load"),
        DeclareLaunchArgument("map_topic", default_value="/map"),
        DeclareLaunchArgument("map_output_dir", default_value="/tmp"),
        DeclareLaunchArgument("map_name", default_value="my_map"),
        DeclareLaunchArgument("map_yaml_path", default_value="/tmp/my_map.yaml"),

        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", new_model_path),

        # 1) Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={
                "world": world,
                "gui": gui,
                "verbose": verbose,
            }.items(),
        ),

        # Static TF: base_footprint -> base_scan (laser)
        # Minimal change: start shortly after spawn to avoid timing races.
        TimerAction(
            period=2.1,
            actions=[
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name="base_to_laser_tf",
                    arguments=["0.0", "0.0", "0.15", "0.0", "0.0", "0.0", "base_footprint", "base_scan"],
                    output="screen",
                ),
            ],
        ),

        # 2) (Optional) robot_state_publisher
        # If you keep this false, you can still navigate if TF is produced by plugins and you don't need the robot model in RViz.
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    condition=do_rsp,
                    package="robot_state_publisher",
                    executable="robot_state_publisher",
                    output="screen",
                    parameters=[use_sim_time_param],
                    # NOTE: You must provide robot_description if use_rsp:=true.
                    # If you want this path, tell me your URDF location and I'll wire it cleanly.
                ),
            ],
        ),

        # 3) Spawn TB3 in Gazebo from SDF
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package="gazebo_ros",
                    executable="spawn_entity.py",
                    output="screen",
                    arguments=[
                        "-entity", robot_name,
                        "-file", tb3_model_sdf,
                        "-x", "-2.0",
                        "-y", "-0.5",
                        "-z", "0.01",
                    ],
                ),
            ],
        ),

        # 4) SLAM + RViz
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package="ekf_slam_ros",
                    executable="slam_plumbing_node",
                    name="slam_plumbing_node",
                    output="screen",
                    parameters=[slam_params, use_sim_time_param],
                ),
                Node(
                    package="rviz2",
                    executable="rviz2",
                    output="screen",
                    parameters=[use_sim_time_param],
                    additional_env={
                        "QT_QPA_PLATFORM": "xcb",
                        "LIBGL_ALWAYS_SOFTWARE": "1",
                    },
                ),
            ],
        ),

        # 5) Map tools (your code)
        # Load mode: publish /map from a YAML file
        TimerAction(
            period=3.2,
            actions=[
                Node(
                    condition=do_map_load,
                    package="amr_map_tools",
                    executable="map_loader_node",
                    name="map_loader_node",
                    output="screen",
                    parameters=[
                        {"yaml_path": map_yaml_path},
                        {"map_topic": map_topic},
                        use_sim_time_param,
                    ],
                ),
            ],
        ),

        # Save mode: subscribe to /map and expose /save_map service
        TimerAction(
            period=3.2,
            actions=[
                Node(
                    condition=do_map_save,
                    package="amr_map_tools",
                    executable="map_saver_node",
                    name="map_saver_node",
                    output="screen",
                    parameters=[
                        {"map_topic": map_topic},
                        {"output_dir": map_output_dir},
                        {"map_name": map_name},
                        use_sim_time_param,
                    ],
                ),
            ],
        ),
        
                # 5) Global Planner + Pure Pursuit Controller
        TimerAction(
            period=3.5,
            actions=[
                Node(
                    package="amr_navigation",
                    executable="global_planner_node",
                    name="amr_global_planner",
                    output="screen",
                    parameters=[
                        {"use_sim_time": True},
                        {"start_x": -2.0},
                        {"start_y": -0.5},
                    ],
                ),
                Node(
                    package="amr_navigation",
                    executable="amr_controller",
                    name="amr_controller",
                    output="screen",
                    parameters=[
                        {"use_sim_time": True},
                    ],
                ),
            ],
        ),
    ])
