#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


@dataclass
class Pose2D:
    x: float
    y: float
    th: float


class NavState(str, Enum):
    IDLE = "IDLE"
    ALIGN = "ALIGN"
    TRACK = "TRACK"
    RECOVERY_SPIN = "RECOVERY_SPIN"
    RECOVERY_ESCAPE = "RECOVERY_ESCAPE"
    GOAL_REACHED = "GOAL_REACHED"


class AMRController(Node):
    """
    Production-style project controller for a custom AMR stack.

    Inputs:
      - /global_path   (Path, map frame)
      - /slam_pose     (PoseStamped, map frame)
      - /odom          (velocity feedback)
      - /scan          (LaserScan)
      - /goal_pose     (PoseStamped, map frame)

    Output:
      - /cmd_vel
    """

    def __init__(self):
        super().__init__("dwa_controller")

        # Topics
        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("slam_pose_topic", "/slam_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # Motion
        self.declare_parameter("v_nominal", 0.18)
        self.declare_parameter("v_slow", 0.08)
        self.declare_parameter("v_escape", -0.10)
        self.declare_parameter("w_max", 1.2)
        self.declare_parameter("rotate_gain", 1.8)
        self.declare_parameter("track_gain", 1.6)

        # Geometry / safety
        self.declare_parameter("robot_radius", 0.16)
        self.declare_parameter("safety_margin", 0.05)
        self.declare_parameter("front_obstacle_stop_dist", 0.32)
        self.declare_parameter("front_obstacle_slow_dist", 0.55)
        self.declare_parameter("side_sector_deg", 65.0)
        self.declare_parameter("front_sector_deg", 28.0)

        # Path following
        self.declare_parameter("lookahead_dist", 0.65)
        self.declare_parameter("goal_tolerance", 0.18)
        self.declare_parameter("heading_align_threshold", 0.55)
        self.declare_parameter("heading_track_threshold", 0.22)

        # Progress / stuck detection
        self.declare_parameter("progress_timeout_sec", 2.0)
        self.declare_parameter("progress_min_distance", 0.06)
        self.declare_parameter("stuck_speed_threshold", 0.02)
        self.declare_parameter("stuck_cycles_before_recovery", 18)

        # Recovery timing
        self.declare_parameter("spin_recovery_time", 1.8)
        self.declare_parameter("escape_recovery_time", 2.2)
        self.declare_parameter("recovery_cooldown_time", 1.0)

        # Control
        self.declare_parameter("control_period", 0.05)

        self.path: Optional[Path] = None
        self.slam_pose: Optional[PoseStamped] = None
        self.odom: Optional[Odometry] = None
        self.scan: Optional[LaserScan] = None
        self.goal_pose: Optional[PoseStamped] = None

        self.path_pts: List[Tuple[float, float]] = []
        self.closest_idx = 0

        self.state = NavState.IDLE
        self.recovery_end_time: Optional[float] = None
        self.recovery_cooldown_until: float = 0.0
        self.escape_turn_sign: float = 1.0

        self.stuck_counter = 0
        self.last_progress_pose: Optional[Pose2D] = None
        self.last_progress_time: float = 0.0

        self.create_subscription(Path, self.get_parameter("path_topic").value, self._path_cb, 10)
        self.create_subscription(PoseStamped, self.get_parameter("slam_pose_topic").value, self._slam_pose_cb, 20)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._odom_cb, 50)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._scan_cb, 20)
        self.create_subscription(PoseStamped, self.get_parameter("goal_topic").value, self._goal_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.timer = self.create_timer(float(self.get_parameter("control_period").value), self._tick)

        self.get_logger().info("AMR controller started.")

    # ---------------- ROS callbacks ----------------
    def _path_cb(self, msg: Path):
        self.path = msg
        self.path_pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.closest_idx = 0
        self.get_logger().info(f"Received global path with {len(self.path_pts)} poses.")

    def _slam_pose_cb(self, msg: PoseStamped):
        self.slam_pose = msg

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal_pose = msg
        self.get_logger().info(
            f"Received goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )
        if self.state in (NavState.IDLE, NavState.GOAL_REACHED):
            self.set_state(NavState.ALIGN)

    # ---------------- Main loop ----------------
    def _tick(self):
        if self.slam_pose is None or self.odom is None or self.scan is None:
            return

        pose = self.get_pose_from_slam(self.slam_pose)
        now = self.now_s()

        if self.goal_pose is None or not self.path_pts:
            self.set_state(NavState.IDLE)
            self.publish_stop()
            return

        if self.goal_reached(pose):
            self.set_state(NavState.GOAL_REACHED)
            self.publish_stop()
            return

        target = self.pick_target_on_path(pose.x, pose.y)
        heading_err = self.angle_to_target(pose, target)

        if self.state == NavState.IDLE:
            self.set_state(NavState.ALIGN)

        if self.state == NavState.ALIGN:
            if abs(heading_err) <= float(self.get_parameter("heading_track_threshold").value):
                self.reset_progress_monitor(pose)
                self.set_state(NavState.TRACK)
            else:
                self.publish_rotate_toward(heading_err)
                return

        if self.state == NavState.TRACK:
            if abs(heading_err) > float(self.get_parameter("heading_align_threshold").value):
                self.set_state(NavState.ALIGN)
                self.publish_rotate_toward(heading_err)
                return

            cmd = self.compute_tracking_cmd(pose, target, heading_err)
            self.cmd_pub.publish(cmd)

            self.update_progress_monitor(pose, cmd, now)

            if now >= self.recovery_cooldown_until and self.is_stuck():
                self.get_logger().warn("No progress detected. Entering recovery.")
                self.start_spin_recovery()
            return

        if self.state == NavState.RECOVERY_SPIN:
            if self.recovery_done(now):
                self.start_escape_recovery()
                return

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.55 * self.escape_turn_sign
            self.cmd_pub.publish(cmd)
            return

        if self.state == NavState.RECOVERY_ESCAPE:
            if self.recovery_done(now):
                self.finish_recovery(pose)
                return

            cmd = Twist()
            cmd.linear.x = float(self.get_parameter("v_escape").value)
            cmd.angular.z = 0.6 * self.escape_turn_sign
            self.cmd_pub.publish(cmd)
            return

        if self.state == NavState.GOAL_REACHED:
            self.publish_stop()

    # ---------------- State / recovery ----------------
    def set_state(self, new_state: NavState):
        if self.state != new_state:
            self.state = new_state
            self.get_logger().info(f"State -> {self.state}")

    def start_spin_recovery(self):
        self.escape_turn_sign = self.choose_escape_turn_sign()
        self.set_state(NavState.RECOVERY_SPIN)
        self.recovery_end_time = self.now_s() + float(self.get_parameter("spin_recovery_time").value)

    def start_escape_recovery(self):
        self.set_state(NavState.RECOVERY_ESCAPE)
        self.recovery_end_time = self.now_s() + float(self.get_parameter("escape_recovery_time").value)

    def finish_recovery(self, pose: Pose2D):
        self.stuck_counter = 0
        self.recovery_end_time = None
        self.recovery_cooldown_until = self.now_s() + float(self.get_parameter("recovery_cooldown_time").value)
        self.reset_progress_monitor(pose)
        self.set_state(NavState.ALIGN)

    def recovery_done(self, now_s: float) -> bool:
        return self.recovery_end_time is None or now_s >= self.recovery_end_time

    # ---------------- Pose / path helpers ----------------
    def get_pose_from_slam(self, pose_msg: PoseStamped) -> Pose2D:
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        return Pose2D(float(p.x), float(p.y), yaw_from_quat(q.x, q.y, q.z, q.w))

    def goal_reached(self, pose: Pose2D) -> bool:
        gx = float(self.goal_pose.pose.position.x)
        gy = float(self.goal_pose.pose.position.y)
        return math.hypot(gx - pose.x, gy - pose.y) <= float(self.get_parameter("goal_tolerance").value)

    def angle_to_target(self, pose: Pose2D, target: Tuple[float, float]) -> float:
        tx, ty = target
        desired = math.atan2(ty - pose.y, tx - pose.x)
        return wrap(desired - pose.th)

    def pick_target_on_path(self, x: float, y: float) -> Tuple[float, float]:
        self.closest_idx = self.find_closest_index(self.path_pts, x, y, self.closest_idx)
        L = float(self.get_parameter("lookahead_dist").value)

        for i in range(self.closest_idx, len(self.path_pts)):
            px, py = self.path_pts[i]
            if math.hypot(px - x, py - y) >= L:
                return (px, py)

        return self.path_pts[-1]

    def find_closest_index(self, pts, x, y, start_idx: int) -> int:
        best_i = start_idx
        best_d2 = float("inf")
        end = min(len(pts), start_idx + 250)
        for i in range(start_idx, end):
            dx = pts[i][0] - x
            dy = pts[i][1] - y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    # ---------------- Tracking ----------------
    def compute_tracking_cmd(self, pose: Pose2D, target: Tuple[float, float], heading_err: float) -> Twist:
        front_clearance = self.front_clearance(self.scan)

        v_nominal = float(self.get_parameter("v_nominal").value)
        v_slow = float(self.get_parameter("v_slow").value)
        w_max = float(self.get_parameter("w_max").value)
        track_gain = float(self.get_parameter("track_gain").value)

        stop_dist = float(self.get_parameter("front_obstacle_stop_dist").value)
        slow_dist = float(self.get_parameter("front_obstacle_slow_dist").value)

        cmd = Twist()

        if front_clearance < stop_dist:
            cmd.linear.x = 0.0
            cmd.angular.z = max(-w_max, min(w_max, track_gain * heading_err))
            return cmd

        if front_clearance < slow_dist or abs(heading_err) > 0.35:
            cmd.linear.x = v_slow
        else:
            cmd.linear.x = v_nominal

        cmd.angular.z = max(-w_max, min(w_max, track_gain * heading_err))
        return cmd

    def publish_rotate_toward(self, heading_err: float):
        w_max = float(self.get_parameter("w_max").value)
        gain = float(self.get_parameter("rotate_gain").value)

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = max(-w_max, min(w_max, gain * heading_err))
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    # ---------------- Scan analysis ----------------
    def front_clearance(self, scan: LaserScan) -> float:
        front_deg = float(self.get_parameter("front_sector_deg").value)
        return self.min_clearance_in_sector(scan, -front_deg, front_deg)

    def side_clearances(self, scan: LaserScan) -> Tuple[float, float]:
        side_deg = float(self.get_parameter("side_sector_deg").value)
        left = self.avg_clearance_in_sector(scan, 20.0, side_deg)
        right = self.avg_clearance_in_sector(scan, -side_deg, -20.0)
        return left, right

    def choose_escape_turn_sign(self) -> float:
        left, right = self.side_clearances(self.scan)
        sign = 1.0 if left >= right else -1.0
        self.get_logger().info(
            f"Recovery choosing {'left' if sign > 0 else 'right'} escape. left={left:.2f}, right={right:.2f}"
        )
        return sign

    def min_clearance_in_sector(self, scan: LaserScan, deg_min: float, deg_max: float) -> float:
        if scan is None or not scan.ranges:
            return 10.0

        a = scan.angle_min
        best = float("inf")
        amin = math.radians(deg_min)
        amax = math.radians(deg_max)

        for r in scan.ranges:
            if amin <= a <= amax and math.isfinite(r) and scan.range_min < r < scan.range_max:
                if r < best:
                    best = r
            a += scan.angle_increment

        return best if math.isfinite(best) else 10.0

    def avg_clearance_in_sector(self, scan: LaserScan, deg_min: float, deg_max: float) -> float:
        if scan is None or not scan.ranges:
            return 10.0

        a = scan.angle_min
        vals = []
        amin = math.radians(deg_min)
        amax = math.radians(deg_max)

        low = min(amin, amax)
        high = max(amin, amax)

        for r in scan.ranges:
            if low <= a <= high and math.isfinite(r) and scan.range_min < r < scan.range_max:
                vals.append(r)
            a += scan.angle_increment

        if not vals:
            return 10.0
        return float(sum(vals) / len(vals))

    # ---------------- Stuck / progress monitoring ----------------
    def reset_progress_monitor(self, pose: Pose2D):
        self.last_progress_pose = Pose2D(pose.x, pose.y, pose.th)
        self.last_progress_time = self.now_s()
        self.stuck_counter = 0

    def update_progress_monitor(self, pose: Pose2D, cmd: Twist, now_s: float):
        actual_v = float(self.odom.twist.twist.linear.x)
        actual_w = float(self.odom.twist.twist.angular.z)
        stuck_speed = float(self.get_parameter("stuck_speed_threshold").value)

        commanded_motion = abs(cmd.linear.x) > 0.04 or abs(cmd.angular.z) > 0.12
        actual_motion = abs(actual_v) > stuck_speed or abs(actual_w) > stuck_speed

        if commanded_motion and not actual_motion:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        if self.last_progress_pose is None:
            self.reset_progress_monitor(pose)
            return

        progress = math.hypot(
            pose.x - self.last_progress_pose.x,
            pose.y - self.last_progress_pose.y
        )

        timeout = float(self.get_parameter("progress_timeout_sec").value)
        min_dist = float(self.get_parameter("progress_min_distance").value)

        if progress >= min_dist:
            self.last_progress_pose = Pose2D(pose.x, pose.y, pose.th)
            self.last_progress_time = now_s
            self.stuck_counter = 0
        elif now_s - self.last_progress_time > timeout:
            self.stuck_counter += 2

    def is_stuck(self) -> bool:
        return self.stuck_counter >= int(self.get_parameter("stuck_cycles_before_recovery").value)

    # ---------------- Time ----------------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main():
    rclpy.init()
    node = AMRController()
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
