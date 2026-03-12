import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/nihal/amr_ws/src/ekf_slam_ros/install/ekf_slam_ros'
