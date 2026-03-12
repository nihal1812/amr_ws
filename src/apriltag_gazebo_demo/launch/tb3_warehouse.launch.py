from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def _join_paths(*paths):
    return ":".join([p for p in paths if p])


def generate_launch_description():
    bringup_share = get_package_share_directory('apriltag_gazebo_demo')
    world_default = os.path.join(bringup_share, 'worlds', 'warehouse_with_tags.world')

    gazebo_launch = os.path.join(
        get_package_share_directory('gazebo_ros'),
        'launch',
        'gazebo.launch.py'
    )
    
    gzserver_launch = os.path.join(
        get_package_share_directory('gazebo_ros'),
        'launch', 'gzserver.launch.py'
    )

    # TB3 model SDF (VERIFY this path via the find command I gave you)
    tb3_model_sdf = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models', 'turtlebot3_waffle_pi', 'model.sdf'
    )

    # Model paths
    my_models = os.path.join(bringup_share, 'models')

    aws_root = get_package_share_directory('aws_robomaker_small_warehouse_world')
    aws_models = os.path.join(aws_root, 'models')

    # Launch args
    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    verbose = LaunchConfiguration('verbose')

    # Env path appends (safe)
    old_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    new_model_path = f"{os.environ.get('GAZEBO_MODEL_PATH','')}:{my_models}:{aws_models}"

    old_resource_path = os.environ.get('GAZEBO_RESOURCE_PATH', '')
    # AWS root contains "models/..." and other resources; this helps with file://models/... references
    new_resource_path = _join_paths(old_resource_path, aws_root)

    # AprilTag: change these 2 remaps if your camera topics are different
    image_topic = '/camera/image_raw'
    camera_info_topic = '/camera/camera_info'

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=world_default),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('verbose', default_value='false'),

        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle_pi'),

        # Optional but usually helpful
        SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb'),
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),

        SetEnvironmentVariable('GAZEBO_MODEL_PATH', new_model_path),
        SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', '/usr/share/gazebo-11'),

        # 1) Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={
                'world': world,
                'gui': gui,
                'verbose': verbose,
            }.items()
        ),

        # 2) Spawn TB3 after a short delay (prevents race conditions)
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-entity', 'waffle_pi',
                        '-file', tb3_model_sdf,
                        '-x', '-2.0',
                        '-y', '-0.5',
                        '-z', '0.01',
                    ],
                    output='screen'
                ),
            ]
        ),

        # 3) AprilTag detector (starts after spawn delay too)
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='apriltag_ros',
                    executable='apriltag_node',
                    name='apriltag_node',
                    output='screen',
                    parameters=[{
                        'tag_family': '36h11',
                        'tag_size': 0.20,          # must match your SDF tag size (0.2m)
                        'max_hamming': 0,
                    }],
                    remappings=[
                        ('image_rect', '/camera/image_raw'),
                        ('camera_info', '/camera/camera_info'),
                    ]
                )
            ]
        ),
    ])
