"""One outdoor_navigation grid_mapper per env_det group cloud (needs the grid_mapper package, e.g. in
lero_isaac_stack's devcontainer, where it is built):

    ros2 launch env_det_grid_mapper.launch.py [groups:=stop_points,avoid_points] [use_sim_time:=true]
        [namespace:=/perception/env_det] [params_file:=<this folder>/env_det_grid_mapper.yaml]

Instance `env_det_<group>_grid_mapper` reads <namespace>/<group> and publishes the grid map on
<namespace>/<group>_grid_map (grid_map_msgs/GridMap), e.g. /perception/env_det/stop_points_grid_map.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent


def nodes(context):
    namespace = LaunchConfiguration('namespace').perform(context).rstrip('/')
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    params_file = LaunchConfiguration('params_file').perform(context)
    out = []
    for group in filter(None, LaunchConfiguration('groups').perform(context).split(',')):
        group = group.strip()
        out.append(Node(
            package='grid_mapper', executable='grid_mapper_node', name=f'env_det_{group}_grid_mapper',
            output='both', respawn=True,
            parameters=[params_file, {'pc_input.topic': f'{namespace}/{group}',
                                      'map.topic': f'{namespace}/{group}_grid_map',
                                      'use_sim_time': use_sim_time}]))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('groups', default_value='stop_points,avoid_points'),
        DeclareLaunchArgument('namespace', default_value='/perception/env_det'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('params_file', default_value=str(HERE / 'env_det_grid_mapper.yaml')),
        OpaqueFunction(function=nodes),
    ])
