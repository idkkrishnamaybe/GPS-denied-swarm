"""ArUco "GPS" on the mother drone (Joey), plus the Gazebo camera bridge it needs.

    ros2 launch robofest aruco_localizer.launch.py                       # simulation
    ros2 launch robofest aruco_localizer.launch.py publish_mavros:=true  # feed /NAME/mavros/vision_pose/pose

On the real drone set sim_bridge:=false and point image/camera_info at the real camera driver.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CAM = '/world/iris_runway/model/Joey/model/gimbal/link/pitch_link/sensor/camera'


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('publish_mavros', default_value='false',
                              description='also publish each pose to /NAME/mavros/vision_pose/pose'),
        DeclareLaunchArgument('sim_bridge', default_value='true',
                              description='bridge the Gazebo camera topics into ROS 2'),
        DeclareLaunchArgument('image_topic', default_value=f'{CAM}/image'),
        DeclareLaunchArgument('camera_info_topic', default_value=f'{CAM}/camera_info'),
        Node(
            package='ros_gz_bridge', executable='parameter_bridge', name='mother_camera_bridge',
            condition=IfCondition(LaunchConfiguration('sim_bridge')),
            arguments=[f'{CAM}/image@sensor_msgs/msg/Image[gz.msgs.Image',
                       f'{CAM}/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'],
            output='screen'),
        Node(
            package='robofest', executable='aruco_localizer', output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'dictionary': 'DICT_5X5_100',
                'marker_size': 0.18,                  # black square edge, metres
                'markers': ['10:Joey', '11:DeeDee', '12:Marky'],
                'reference_id': 12,                   # the landed drone
                'self_name': 'Joey',                  # the drone carrying this camera
                'reference_marker_height': 0.331,     # reference marker above ground when landed
                'marker_offset_z': 0.136,             # marker above each drone's body origin
                'camera_offset': [0.0, -0.01, -0.1249],
                'attitude_topic': '/{self}/mavros/imu/data',
                'publish_mavros': LaunchConfiguration('publish_mavros'),
            }]),
    ])
