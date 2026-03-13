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
    def __init__(self):
        super().__init__("amr_controller")

        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("slam_pose_topic", "/slam_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("v_nominal", 0.20)
        self.declare_parameter("v_slow", 0.10)
        self.declare_parameter("v_escape", -0.10)
        self.declare_parameter("w_max", 0.85)
        self.declare_parameter("rotate_gain", 1.35)
        self.declare_parameter("track_gain", 1.15)

        self.declare_parameter("lookahead_dist", 0.80)
        self.declare_parameter("goal_tolerance", 0.18)
        self.declare_parameter("heading_align_threshold", 0.80)
        self.declare_parameter("heading_track_threshold", 0.35)

        self.declare_parameter("front_obstacle_stop_dist", 0.28)
        self.declare_parameter("front_obstacle_slow_dist", 0.48)
        self.declare_parameter("side_sector_deg", 65.0)
        self.declare_parameter("front_sector_deg", 28.0)

        self.declare_parameter("progress_timeout_sec", 3.5)
        self.declare_parameter("progress_min_distance", 0.035)
        self.declare_parameter("stuck_speed_threshold", 0.012)
        self.declare_parameter("stuck_cycles_before_recovery", 34)

        self.declare_parameter("spin_recovery_time", 0.8)
        self.declare_parameter("escape_recovery_time", 1.0)
        self.declare_parameter("recovery_cooldown_time", 2.5)
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
        self.last_failure_reason = ""
        self.last_debug_time = 0.0

        self.create_subscription(Path, self.get_parameter("path_topic").value, self._path_cb, 10)
        self.create_subscription(PoseStamped, self.get_parameter("slam_pose_topic").value, self._slam_pose_cb, 20)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._odom_cb, 50)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._scan_cb, 20)
        self.create_subscription(PoseStamped, self.get_parameter("goal_topic").value, self._goal_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.timer = self.create_timer(float(self.get_parameter("control_period").value), self._tick)

        self.get_logger().info("AMR controller started.")

    def _path_cb(self, msg: Path):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

        if len(pts) < 2:
            self.path = msg
            self.path_pts = pts
            self.closest_idx = 0
            return

        filtered = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - filtered[-1][0], p[1] - filtered[-1][1]) > 0.03:
                filtered.append(p)

        self.path = msg
        self.path_pts = filtered
        self.closest_idx = 0
        self.get_logger().info(f"Received global path with {len(self.path_pts)} filtered poses.")

    def _slam_pose_cb(self, msg: PoseStamped):
        self.slam_pose = msg

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal_pose = msg
        self.stuck_counter = 0
        self.last_failure_reason = ""

        if self.slam_pose is not None:
            pose = self.get_pose_from_slam(self.slam_pose)
            self.reset_progress_monitor(pose)
        else:
            self.last_progress_pose = None
            self.last_progress_time = self.now_s()

        self.get_logger().info(f"Received goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

        self.set_state(NavState.ALIGN)

    def _tick(self):
        if self.slam_pose is None or self.odom is None or self.scan is None:
            return

        pose = self.get_pose_from_slam(self.slam_pose)
        now = self.now_s()

        if self.goal_pose is None or not self.path_pts:
            self.set_failure_reason("no_goal_or_no_path")
            self.set_state(NavState.IDLE)
            self.publish_stop()
            return

        if self.goal_reached(pose):
            self.set_failure_reason("goal_reached")
            self.set_state(NavState.GOAL_REACHED)
            self.publish_stop()
            return

        target = self.pick_target_on_path(pose.x, pose.y)
        heading_err = self.angle_to_target(pose, target)
        front = self.front_clearance(self.scan)
        left, right = self.side_clearances(self.scan)

        if self.state == NavState.IDLE:
            self.set_state(NavState.ALIGN)

        if self.state == NavState.ALIGN:
            if abs(heading_err) <= float(self.get_parameter("heading_track_threshold").value):
                self.reset_progress_monitor(pose)
                self.set_state(NavState.TRACK)
            else:
                self.set_failure_reason("aligning_large_heading_error")
                self.publish_rotate_toward(heading_err)
                self.debug_print(now, heading_err, front, left, right, None)
                return

        if self.state == NavState.TRACK:
            if abs(heading_err) > float(self.get_parameter("heading_align_threshold").value):
                self.set_failure_reason("heading_too_large_back_to_align")
                self.set_state(NavState.ALIGN)
                self.publish_rotate_toward(heading_err)
                self.debug_print(now, heading_err, front, left, right, None)
                return

            cmd = self.compute_tracking_cmd(heading_err, front)
            self.cmd_pub.publish(cmd)

            self.update_progress_monitor(pose, cmd, now)

            if self.state == NavState.TRACK and now >= self.recovery_cooldown_until and self.is_stuck():
                self.set_failure_reason("stuck_no_progress")
                self.get_logger().warn("No forward progress detected. Entering recovery.")
                self.start_spin_recovery()
                self.debug_print(now, heading_err, front, left, right, cmd)
                return

            self.debug_print(now, heading_err, front, left, right, cmd)
            return

        if self.state == NavState.RECOVERY_SPIN:
            if self.recovery_done(now):
                self.start_escape_recovery()
                return

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.55 * self.escape_turn_sign
            self.cmd_pub.publish(cmd)
            self.set_failure_reason("recovery_spin")
            self.debug_print(now, heading_err, front, left, right, cmd)
            return

        if self.state == NavState.RECOVERY_ESCAPE:
            if self.recovery_done(now):
                self.finish_recovery(pose)
                return

            cmd = Twist()
            cmd.linear.x = float(self.get_parameter("v_escape").value)
            cmd.angular.z = 0.45 * self.escape_turn_sign
            self.cmd_pub.publish(cmd)
            self.set_failure_reason("recovery_escape")
            self.debug_print(now, heading_err, front, left, right, cmd)
            return

        if self.state == NavState.GOAL_REACHED:
            self.publish_stop()

    def set_state(self, new_state: NavState):
        if self.state != new_state:
            self.state = new_state
            self.get_logger().info(f"State -> {self.state}")

    def set_failure_reason(self, reason: str):
        if self.last_failure_reason != reason:
            self.last_failure_reason = reason
            self.get_logger().info(f"Reason -> {reason}")

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
        self.set_failure_reason("recovery_finished")
        self.set_state(NavState.ALIGN)

    def recovery_done(self, now_s: float) -> bool:
        return self.recovery_end_time is None or now_s >= self.recovery_end_time

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

    def compute_tracking_cmd(self, heading_err: float, front_clearance: float) -> Twist:
        v_nominal = float(self.get_parameter("v_nominal").value)
        v_slow = float(self.get_parameter("v_slow").value)
        w_max = float(self.get_parameter("w_max").value)
        track_gain = float(self.get_parameter("track_gain").value)

        stop_dist = float(self.get_parameter("front_obstacle_stop_dist").value)
        slow_dist = float(self.get_parameter("front_obstacle_slow_dist").value)

        cmd = Twist()

        if front_clearance < stop_dist:
            cmd.linear.x = 0.0
            cmd.angular.z = max(-0.6, min(0.6, 0.8 * heading_err))
            self.set_failure_reason("front_blocked_rotate_only")
            return cmd

        if front_clearance < slow_dist or abs(heading_err) > 0.35:
            cmd.linear.x = v_slow
        else:
            cmd.linear.x = v_nominal

        cmd.angular.z = max(-w_max, min(w_max, track_gain * heading_err))
        self.set_failure_reason("tracking")
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

    def front_clearance(self, scan: LaserScan) -> float:
        half = float(self.get_parameter("front_sector_deg").value)
        return self.min_clearance_in_sector(scan, -half, half)

    def side_clearances(self, scan: LaserScan) -> Tuple[float, float]:
        side = float(self.get_parameter("side_sector_deg").value)
        left = self.avg_clearance_in_sector(scan, 20.0, side)
        right = self.avg_clearance_in_sector(scan, -side, -20.0)
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
        low = min(amin, amax)
        high = max(amin, amax)

        for r in scan.ranges:
            if low <= a <= high and math.isfinite(r) and scan.range_min < r < scan.range_max:
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

    def reset_progress_monitor(self, pose: Pose2D):
        self.last_progress_pose = Pose2D(pose.x, pose.y, pose.th)
        self.last_progress_time = self.now_s()
        self.stuck_counter = 0

    def update_progress_monitor(self, pose: Pose2D, cmd: Twist, now_s: float):
        actual_v = float(self.odom.twist.twist.linear.x)
        stuck_speed = float(self.get_parameter("stuck_speed_threshold").value)

        if self.last_progress_pose is None:
            self.reset_progress_monitor(pose)
            return

        progress = math.hypot(
            pose.x - self.last_progress_pose.x,
            pose.y - self.last_progress_pose.y
        )

        timeout = float(self.get_parameter("progress_timeout_sec").value)
        min_dist = float(self.get_parameter("progress_min_distance").value)

        trying_to_translate = abs(cmd.linear.x) > 0.05
        actual_translation = abs(actual_v) > stuck_speed

        if progress >= min_dist:
            self.last_progress_pose = Pose2D(pose.x, pose.y, pose.th)
            self.last_progress_time = now_s
            self.stuck_counter = 0
            return

        if trying_to_translate:
            if not actual_translation:
                self.stuck_counter += 1

            if now_s - self.last_progress_time > timeout:
                self.stuck_counter += 1
                self.last_progress_time = now_s
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
            self.last_progress_time = now_s

    def is_stuck(self) -> bool:
        return self.stuck_counter >= int(self.get_parameter("stuck_cycles_before_recovery").value)

    def debug_print(self, now: float, heading_err: float, front: float, left: float, right: float, cmd: Optional[Twist]):
        if now - self.last_debug_time < 0.7:
            return
        self.last_debug_time = now

        if cmd is None:
            v = 0.0
            w = 0.0
        else:
            v = cmd.linear.x
            w = cmd.angular.z

        self.get_logger().info(
            f"[CTRL] state={self.state} reason={self.last_failure_reason} "
            f"heading_err={heading_err:.3f} front={front:.2f} left={left:.2f} right={right:.2f} "
            f"stuck_counter={self.stuck_counter} cmd=({v:.2f},{w:.2f})"
        )

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
