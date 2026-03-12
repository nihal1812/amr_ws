import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster

# Helper functions
def yaw_from_quat(q):
    # General yaw extraction (robust even if small roll/pitch exist)
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )

def quat_from_yaw(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def invert_se2(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)

    x_inv = -c * x - s * y
    y_inv =  s * x - c * y
    yaw_inv = -yaw

    return x_inv, y_inv, yaw_inv


def compose_se2(a, b):
    x1, y1, yaw1 = a
    x2, y2, yaw2 = b

    c = math.cos(yaw1)
    s = math.sin(yaw1)

    x = x1 + c * x2 - s * y2
    y = y1 + s * x2 + c * y2

    # wrap angle
    yaw = math.atan2(
        math.sin(yaw1 + yaw2),
        math.cos(yaw1 + yaw2)
    )

    return x, y, yaw


class SlamPlumbingNode(Node):

    def __init__(self):
        super().__init__('slam_plumbing_node')

        self.pose_pub = self.create_publisher(PoseStamped, '/slam_pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_cb,
            50
        )

        self.get_logger().info("SLAM plumbing node started (computed map->odom)")


    def odom_cb(self, msg: Odometry):

        # --------------------------------------
        # 1) Extract T_odom_base from /odom
        # --------------------------------------
        x_o = msg.pose.pose.position.x
        y_o = msg.pose.pose.position.y
        yaw_o = yaw_from_quat(msg.pose.pose.orientation)

        T_odom_base = (x_o, y_o, yaw_o)

        # --------------------------------------
        # 2) Define T_map_base (for now = odom)
        # --------------------------------------
        # Later this will come from EKF
        T_map_base = (x_o, y_o, yaw_o)

        # --------------------------------------
        # 3) Compute T_map_odom
        # --------------------------------------
        T_base_odom = invert_se2(*T_odom_base)
        T_map_odom = compose_se2(T_map_base, T_base_odom)

        x_mo, y_mo, yaw_mo = T_map_odom

        # --------------------------------------
        # 4) Publish slam pose (map frame)
        # --------------------------------------
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "map"
        out.pose = msg.pose.pose
        self.pose_pub.publish(out)

        # --------------------------------------
        # 5) Broadcast TF map -> odom
        # --------------------------------------
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        t.transform.translation.x = x_mo
        t.transform.translation.y = y_mo
        t.transform.translation.z = 0.0

        t.transform.rotation = quat_from_yaw(yaw_mo)

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = SlamPlumbingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
