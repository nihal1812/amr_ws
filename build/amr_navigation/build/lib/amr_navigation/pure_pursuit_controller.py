#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    # yaw (Z) from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class PurePursuitController(Node):
    """
    Pure pursuit path follower + simple scan-based safety layer.
    """

    def __init__(self):
        super().__init__("pure_pursuit_controller")

        # --- Params ---
        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("lookahead", 0.6)          # meters
        self.declare_parameter("v_nominal", 0.25)         # m/s
        self.declare_parameter("v_min", 0.05)             # m/s
        self.declare_parameter("w_max", 1.2)              # rad/s
        self.declare_parameter("goal_tolerance", 0.20)    # meters

        # Safety
        self.declare_parameter("front_arc_deg", 60.0)     # +/- degrees
        self.declare_parameter("stop_dist", 0.35)         # meters
        self.declare_parameter("slow_dist", 0.60)         # meters

        # Control loop
        self.declare_parameter("control_period", 0.05)    # 20 Hz

        self.path: Optional[Path] = None
        self.odom: Optional[Odometry] = None
        self.scan: Optional[LaserScan] = None

        self._closest_idx = 0

        path_topic = self.get_parameter("path_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        scan_topic = self.get_parameter("scan_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        period = float(self.get_parameter("control_period").value)

        self.create_subscription(Path, path_topic, self._path_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 50)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(f"Pure Pursuit: path={path_topic}, odom={odom_topic}, scan={scan_topic}, cmd={cmd_vel_topic}")

    # ---------- Callbacks ----------
    def _path_cb(self, msg: Path):
        if len(msg.poses) < 2:
            self.get_logger().warn("Received path with <2 poses; ignoring.")
            self.path = None
            return
        self.path = msg
        self._closest_idx = 0

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    # ---------- Core control ----------
    def _tick(self):
        if self.path is None or self.odom is None:
            return

        # 1) Get robot pose (assume odom frame for now)
        px = float(self.odom.pose.pose.position.x)
        py = float(self.odom.pose.pose.position.y)
        q = self.odom.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # 2) Extract path points
        pts = [(p.pose.position.x, p.pose.position.y) for p in self.path.poses]

        # 3) Find closest point index (search forward only for speed)
        self._closest_idx = self.find_closest_index(pts, px, py, self._closest_idx)

        # 4) Pick lookahead point
        Ld = float(self.get_parameter("lookahead").value)
        look_pt, goal_reached = self.find_lookahead_point(pts, px, py, self._closest_idx, Ld)

        # If we are close to final goal, stop
        if goal_reached:
            self.cmd_pub.publish(Twist())
            return

        lx, ly = look_pt

        # 5) Transform lookahead point into robot frame (x_r, y_r)
        dx = lx - px
        dy = ly - py

        # rotation by -yaw
        x_r = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        y_r = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        # If lookahead ends up behind robot, rotate in place to reacquire
        if x_r <= 0.05:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6 * (1.0 if y_r >= 0.0 else -1.0)
            self.cmd_pub.publish(cmd)
            return

        # 6) Curvature and cmd
        L2 = x_r * x_r + y_r * y_r
        kappa = 2.0 * y_r / max(L2, 1e-6)

        v = float(self.get_parameter("v_nominal").value)
        w = v * kappa

        # Clamp angular speed
        w_max = float(self.get_parameter("w_max").value)
        w = max(-w_max, min(w_max, w))

        # 7) Safety modulation using scan (front arc)
        v = self.apply_scan_safety(v)

        # Ensure small forward creep if turning hard (optional)
        v_min = float(self.get_parameter("v_min").value)
        if v > 0.0:
            v = max(v_min, v)

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

    # ---------- Helper methods ----------
    def find_closest_index(self, pts: List[Tuple[float, float]], x: float, y: float, start_idx: int) -> int:
        best_i = start_idx
        best_d2 = float("inf")

        # search window forward (prevents full scan every tick)
        end = min(len(pts), start_idx + 200)
        for i in range(start_idx, end):
            dx = pts[i][0] - x
            dy = pts[i][1] - y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def find_lookahead_point(
        self,
        pts: List[Tuple[float, float]],
        x: float,
        y: float,
        start_idx: int,
        Ld: float
    ) -> Tuple[Tuple[float, float], bool]:
        # goal check
        gx, gy = pts[-1]
        goal_tol = float(self.get_parameter("goal_tolerance").value)
        if math.hypot(gx - x, gy - y) < goal_tol:
            return (gx, gy), True

        # walk forward until distance >= lookahead
        for i in range(start_idx, len(pts)):
            px, py = pts[i]
            if math.hypot(px - x, py - y) >= Ld:
                return (px, py), False

        return pts[-1], False

    def apply_scan_safety(self, v: float) -> float:
        if self.scan is None:
            return v

        stop_dist = float(self.get_parameter("stop_dist").value)
        slow_dist = float(self.get_parameter("slow_dist").value)
        arc_deg = float(self.get_parameter("front_arc_deg").value)

        a_min = self.scan.angle_min
        a_inc = self.scan.angle_increment
        n = len(self.scan.ranges)

        # Front arc: [-arc/2, +arc/2]
        arc = math.radians(arc_deg)
        left = +arc / 2.0
        right = -arc / 2.0

        # Convert angles to indices
        def angle_to_idx(a: float) -> int:
            return int(round((a - a_min) / a_inc))

        i0 = max(0, min(n - 1, angle_to_idx(right)))
        i1 = max(0, min(n - 1, angle_to_idx(left)))
        if i1 < i0:
            i0, i1 = i1, i0

        min_r = float("inf")
        for r in self.scan.ranges[i0:i1 + 1]:
            if math.isfinite(r) and r > 0.01:
                if r < min_r:
                    min_r = r

        if min_r == float("inf"):
            return v

        # Hard stop
        if min_r < stop_dist:
            return 0.0

        # Slow down linearly between slow_dist and stop_dist
        if min_r < slow_dist:
            # scale in [0..1]
            alpha = (min_r - stop_dist) / max(slow_dist - stop_dist, 1e-6)
            alpha = max(0.0, min(1.0, alpha))
            return v * alpha

        return v


def main():
    rclpy.init()
    node = PurePursuitController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
