#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


@dataclass
class Pose2D:
    x: float
    y: float
    th: float


class DWAController(Node):
    """
    DWA local planner + controller (outputs cmd_vel)

    Uses:
      - /global_path   (Path in map frame)
      - /slam_pose     (PoseStamped in map frame)
      - /odom          (for current v, w only)
      - /scan          (LaserScan)

    Important:
      - Pose and path are both in map frame now.
      - This removes the old map-vs-odom mismatch.
    """

    def __init__(self):
        super().__init__("dwa_controller")

        # Topics
        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("slam_pose_topic", "/slam_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("goal_topic", "/goal_pose")

        # Robot limits
        self.declare_parameter("v_max", 0.35)
        self.declare_parameter("v_min", 0.0)
        self.declare_parameter("w_max", 1.5)
        self.declare_parameter("a_v", 0.8)
        self.declare_parameter("a_w", 2.5)

        # Sampling
        self.declare_parameter("num_v_samples", 8)
        self.declare_parameter("num_w_samples", 15)

        # Simulation
        self.declare_parameter("sim_dt", 0.05)
        self.declare_parameter("sim_horizon", 1.5)

        # Robot footprint
        self.declare_parameter("robot_radius", 0.18)
        self.declare_parameter("obstacle_margin", 0.05)

        # Costs
        self.declare_parameter("w_path", 1.2)
        self.declare_parameter("w_goal", 1.0)
        self.declare_parameter("w_obs", 2.5)
        self.declare_parameter("w_smooth", 0.15)
        self.declare_parameter("w_speed", 0.3)

        # Guidance
        self.declare_parameter("lookahead_along_path", 0.8)
        self.declare_parameter("goal_tolerance", 0.20)
        self.declare_parameter("yaw_goal_tolerance", 0.20)

        # Recovery / behavior
        self.declare_parameter("rotate_in_place_speed", 0.35)
        self.declare_parameter("min_forward_speed", 0.03)

        # Control loop
        self.declare_parameter("control_period", 0.05)

        self.path: Optional[Path] = None
        self.slam_pose: Optional[PoseStamped] = None
        self.odom: Optional[Odometry] = None
        self.scan: Optional[LaserScan] = None
        self.goal_pose: Optional[PoseStamped] = None

        self._path_pts: List[Tuple[float, float]] = []
        self._closest_idx = 0

        self.create_subscription(Path, self.get_parameter("path_topic").value, self._path_cb, 10)
        self.create_subscription(PoseStamped, self.get_parameter("slam_pose_topic").value, self._slam_pose_cb, 20)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._odom_cb, 50)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._scan_cb, 10)
        self.create_subscription(PoseStamped, self.get_parameter("goal_topic").value, self._goal_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.timer = self.create_timer(float(self.get_parameter("control_period").value), self._tick)

        self.get_logger().info("DWA controller started.")

    # ---------------- Callbacks ----------------
    def _path_cb(self, msg: Path):
        if len(msg.poses) < 2:
            self.path = None
            self._path_pts = []
            return
        self.path = msg
        self._path_pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._closest_idx = 0

    def _slam_pose_cb(self, msg: PoseStamped):
        self.slam_pose = msg

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal_pose = msg

    # ---------------- Main loop ----------------
    def _tick(self):
        if self.slam_pose is None or self.odom is None or self.scan is None:
            return

        if not self._path_pts:
            self.cmd_pub.publish(Twist())
            return

        pose = self.get_pose_from_slam(self.slam_pose)

        if self.goal_pose is not None:
            gx = float(self.goal_pose.pose.position.x)
            gy = float(self.goal_pose.pose.position.y)
            goal_dist = math.hypot(gx - pose.x, gy - pose.y)
            if goal_dist <= float(self.get_parameter("goal_tolerance").value):
                self.cmd_pub.publish(Twist())
                return

        target = self.pick_target_on_path(pose.x, pose.y)

        v0 = float(self.odom.twist.twist.linear.x)
        w0 = float(self.odom.twist.twist.angular.z)

        obs_pts = self.scan_to_points_robot_frame(self.scan)

        v_min, v_max, w_min, w_max = self.dynamic_window(v0, w0)

        best = None
        best_cost = float("inf")

        v_samples, w_samples = self.sample_window(v_min, v_max, w_min, w_max)

        for v in v_samples:
            for w in w_samples:
                traj = self.rollout(pose, v, w)
                cost = self.evaluate(traj, v, w, target, obs_pts, pose)
                if cost < best_cost:
                    best_cost = cost
                    best = (v, w)

        if best is None or not math.isfinite(best_cost):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = float(self.get_parameter("rotate_in_place_speed").value)
            self.cmd_pub.publish(cmd)
            return

        cmd = Twist()
        cmd.linear.x = float(best[0])
        cmd.angular.z = float(best[1])

        self.get_logger().debug(
            f"best v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}, cost={best_cost:.3f}"
        )

        self.cmd_pub.publish(cmd)

    # ---------------- Pose / target selection ----------------
    def get_pose_from_slam(self, pose_msg: PoseStamped) -> Pose2D:
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        return Pose2D(float(p.x), float(p.y), float(yaw))

    def pick_target_on_path(self, x: float, y: float) -> Tuple[float, float]:
        pts = self._path_pts
        self._closest_idx = self.find_closest_index(pts, x, y, self._closest_idx)

        L = float(self.get_parameter("lookahead_along_path").value)
        for i in range(self._closest_idx, len(pts)):
            px, py = pts[i]
            if math.hypot(px - x, py - y) >= L:
                return (px, py)
        return pts[-1]

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

    # ---------------- DWA pieces ----------------
    def dynamic_window(self, v0: float, w0: float):
        v_max_abs = float(self.get_parameter("v_max").value)
        v_min_abs = float(self.get_parameter("v_min").value)
        w_max_abs = float(self.get_parameter("w_max").value)
        a_v = float(self.get_parameter("a_v").value)
        a_w = float(self.get_parameter("a_w").value)
        dt = float(self.get_parameter("control_period").value)

        v_min = max(v_min_abs, v0 - a_v * dt)
        v_max = min(v_max_abs, v0 + a_v * dt)
        w_min = max(-w_max_abs, w0 - a_w * dt)
        w_max = min(+w_max_abs, w0 + a_w * dt)

        return v_min, v_max, w_min, w_max

    def sample_window(self, v_min: float, v_max: float, w_min: float, w_max: float):
        num_v = max(2, int(self.get_parameter("num_v_samples").value))
        num_w = max(3, int(self.get_parameter("num_w_samples").value))
        min_forward = float(self.get_parameter("min_forward_speed").value)

        if v_max < v_min:
            v_max = v_min
        if w_max < w_min:
            w_max = w_min

        if abs(v_max - v_min) < 1e-6:
            v_samples = [v_min]
        else:
            v_samples = [v_min + (v_max - v_min) * i / (num_v - 1) for i in range(num_v)]

        if abs(w_max - w_min) < 1e-6:
            w_samples = [w_min]
        else:
            w_samples = [w_min + (w_max - w_min) * i / (num_w - 1) for i in range(num_w)]

        # Ensure some forward candidates exist at startup
        if v_max > min_forward and all(v < min_forward for v in v_samples):
            v_samples.append(min_forward)

        v_samples = sorted(set(float(v) for v in v_samples))
        w_samples = sorted(set(float(w) for w in w_samples))
        return v_samples, w_samples

    def rollout(self, pose: Pose2D, v: float, w: float) -> List[Pose2D]:
        sim_dt = float(self.get_parameter("sim_dt").value)
        horizon = float(self.get_parameter("sim_horizon").value)
        steps = max(1, int(horizon / sim_dt))

        x, y, th = pose.x, pose.y, pose.th
        traj = []
        for _ in range(steps):
            x += v * math.cos(th) * sim_dt
            y += v * math.sin(th) * sim_dt
            th = wrap(th + w * sim_dt)
            traj.append(Pose2D(x, y, th))
        return traj

    def evaluate(
        self,
        traj: List[Pose2D],
        v: float,
        w: float,
        target: Tuple[float, float],
        obs_pts_robot: List[Tuple[float, float]],
        current_pose: Pose2D,
    ) -> float:
        end = traj[-1]
        tx, ty = target

        d_goal = math.hypot(end.x - tx, end.y - ty)
        d_path = self.distance_to_path(end.x, end.y)

        clearance = self.min_clearance_along_traj(traj, obs_pts_robot, current_pose)
        if clearance <= 0.0:
            return float("inf")

        c_obs = 1.0 / max(clearance, 1e-3)
        smooth = abs(w)
        speed_reward = -v

        w_path = float(self.get_parameter("w_path").value)
        w_goal = float(self.get_parameter("w_goal").value)
        w_obs = float(self.get_parameter("w_obs").value)
        w_smooth = float(self.get_parameter("w_smooth").value)
        w_speed = float(self.get_parameter("w_speed").value)

        return (
            w_path * d_path
            + w_goal * d_goal
            + w_obs * c_obs
            + w_smooth * smooth
            + w_speed * speed_reward
        )

    def distance_to_path(self, x: float, y: float) -> float:
        pts = self._path_pts
        start = max(0, self._closest_idx - 30)
        end = min(len(pts), self._closest_idx + 200)
        best = float("inf")
        for i in range(start, end):
            dx = pts[i][0] - x
            dy = pts[i][1] - y
            d = math.hypot(dx, dy)
            if d < best:
                best = d
        return best if best < float("inf") else 0.0

    # ---------------- Obstacle handling ----------------
    def scan_to_points_robot_frame(self, scan: LaserScan) -> List[Tuple[float, float]]:
        pts = []
        a = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                x = r * math.cos(a)
                y = r * math.sin(a)
                pts.append((x, y))
            a += scan.angle_increment
        return pts

    def min_clearance_along_traj(
        self,
        traj: List[Pose2D],
        obs_pts_robot: List[Tuple[float, float]],
        current_pose: Pose2D,
    ) -> float:
        """
        Approximate clearance check:
        - obstacles are from the current scan in current robot frame
        - convert each simulated pose into current robot frame
        - measure obstacle distance relative to simulated robot center
        """
        R = float(self.get_parameter("robot_radius").value)
        margin = float(self.get_parameter("obstacle_margin").value)

        if not obs_pts_robot:
            return 10.0

        min_clear = float("inf")

        c0 = math.cos(-current_pose.th)
        s0 = math.sin(-current_pose.th)

        sample_idx = sorted(set([0, len(traj) // 3, 2 * len(traj) // 3, len(traj) - 1]))

        for k in sample_idx:
            p = traj[k]

            dx_world = p.x - current_pose.x
            dy_world = p.y - current_pose.y

            # transform simulated center into current robot frame
            dx_robot = c0 * dx_world - s0 * dy_world
            dy_robot = s0 * dx_world + c0 * dy_world

            for ox, oy in obs_pts_robot:
                d = math.hypot(ox - dx_robot, oy - dy_robot) - (R + margin)
                if d < min_clear:
                    min_clear = d

        return float(min_clear)


def main():
    rclpy.init()
    node = DWAController()
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
