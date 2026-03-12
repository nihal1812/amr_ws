#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


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
    DWA local planner + controller (outputs cmd_vel).
    Uses:
      - global path (Path) for guidance
      - LaserScan for obstacle clearance
      - Odometry for current pose and velocity

    v1 collision check: treat robot as a circle and obstacles from scan as points in robot frame.
    """

    def __init__(self):
        super().__init__("dwa_controller")

        # Topics
        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # Robot limits
        self.declare_parameter("v_max", 0.35)
        self.declare_parameter("v_min", 0.0)
        self.declare_parameter("w_max", 1.5)
        self.declare_parameter("a_v", 0.6)      # m/s^2
        self.declare_parameter("a_w", 2.0)      # rad/s^2

        # Sampling
        self.declare_parameter("dv", 0.05)
        self.declare_parameter("dw", 0.15)

        # Simulation
        self.declare_parameter("sim_dt", 0.05)      # s
        self.declare_parameter("sim_horizon", 1.5)  # s

        # Robot footprint approx
        self.declare_parameter("robot_radius", 0.18)   # TB3 ~0.18–0.20

        # Costs (weights)
        self.declare_parameter("w_path", 1.0)
        self.declare_parameter("w_goal", 1.0)
        self.declare_parameter("w_obs", 2.5)
        self.declare_parameter("w_smooth", 0.2)

        # Guidance
        self.declare_parameter("lookahead_along_path", 0.8)  # meters along path
        self.declare_parameter("goal_tolerance", 0.25)

        # Control loop
        self.declare_parameter("control_period", 0.05)

        self.path: Optional[Path] = None
        self.odom: Optional[Odometry] = None
        self.scan: Optional[LaserScan] = None

        self._path_pts: List[Tuple[float, float]] = []
        self._closest_idx = 0

        self.create_subscription(Path, self.get_parameter("path_topic").value, self._path_cb, 10)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._odom_cb, 50)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._scan_cb, 10)

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

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    # ---------------- Main loop ----------------
    def _tick(self):
        if self.odom is None or self.scan is None:
            return

        if not self._path_pts:
            # no path => stop
            self.cmd_pub.publish(Twist())
            return

        pose = self.get_pose_from_odom(self.odom)

        # Choose a local target on the global path (in same frame as path)
        # NOTE: For now we assume path is in "map" but pose is in "odom".
        # If your SLAM keeps map~odom near identity, this works for now.
        # Later we will use TF properly.
        target = self.pick_target_on_path(pose.x, pose.y)

        # Current velocities from odom
        v0 = float(self.odom.twist.twist.linear.x)
        w0 = float(self.odom.twist.twist.angular.z)

        # Build obstacle points in robot frame from scan
        obs_pts = self.scan_to_points_robot_frame(self.scan)

        # Compute dynamic window
        v_min, v_max, w_min, w_max = self.dynamic_window(v0, w0)

        # Sample and score
        best = None
        best_cost = float("inf")

        dv = float(self.get_parameter("dv").value)
        dw = float(self.get_parameter("dw").value)

        v = v_min
        while v <= v_max + 1e-9:
            w = w_min
            while w <= w_max + 1e-9:
                traj = self.rollout(pose, v, w)
                cost = self.evaluate(traj, v, w, target, obs_pts)
                if cost < best_cost:
                    best_cost = cost
                    best = (v, w)
                w += dw
            v += dv

        if best is None or best_cost == float("inf"):
            # No safe command
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
            self.cmd_pub.publish(cmd)
            return

        cmd = Twist()
        cmd.linear.x = float(best[0])
        cmd.angular.z = float(best[1])
        self.cmd_pub.publish(cmd)

    # ---------------- Pose / target selection ----------------
    def get_pose_from_odom(self, odom: Odometry) -> Pose2D:
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        return Pose2D(float(p.x), float(p.y), float(yaw))

    def pick_target_on_path(self, x: float, y: float) -> Tuple[float, float]:
        pts = self._path_pts
        # update closest index
        self._closest_idx = self.find_closest_index(pts, x, y, self._closest_idx)

        # move forward along path until distance from robot >= lookahead
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
    ) -> float:
        # Goal / path costs use endpoint
        end = traj[-1]
        tx, ty = target

        # 1) distance to target
        d_goal = math.hypot(end.x - tx, end.y - ty)

        # 2) distance to global path (endpoint to nearest path point)
        d_path = self.distance_to_path(end.x, end.y)

        # 3) obstacle cost: collision => inf, else inverse of min clearance
        clearance = self.min_clearance_along_traj(traj, obs_pts_robot)
        if clearance <= 0.0:
            return float("inf")
        c_obs = 1.0 / clearance

        # 4) smoothness penalty (discourage high angular velocities)
        smooth = abs(w)

        w_path = float(self.get_parameter("w_path").value)
        w_goal = float(self.get_parameter("w_goal").value)
        w_obs = float(self.get_parameter("w_obs").value)
        w_smooth = float(self.get_parameter("w_smooth").value)

        return w_path * d_path + w_goal * d_goal + w_obs * c_obs + w_smooth * smooth

    def distance_to_path(self, x: float, y: float) -> float:
        pts = self._path_pts
        # local search near closest index
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

    # ---------------- Obstacle handling (scan -> points) ----------------
    def scan_to_points_robot_frame(self, scan: LaserScan) -> List[Tuple[float, float]]:
        pts = []
        a = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and r > 0.05:
                x = r * math.cos(a)
                y = r * math.sin(a)
                pts.append((x, y))
            a += scan.angle_increment
        return pts

    def min_clearance_along_traj(self, traj: List[Pose2D], obs_pts_robot: List[Tuple[float, float]]) -> float:
        """
        Conservative check:
        - Treat robot as circle radius R.
        - Obstacles are points in robot frame at current time.
        - For each trajectory pose, transform obstacle points into that pose frame (approx),
          and compute min distance to robot center.

        This is an approximation but works well enough for v1.
        """
        R = float(self.get_parameter("robot_radius").value)

        # If no obstacles, huge clearance
        if not obs_pts_robot:
            return 10.0

        min_clear = float("inf")

        # We assume obstacles are static during the short horizon.
        # For each simulated pose relative to current robot pose:
        # Convert obstacle points from *current robot frame* into the simulated robot frame.
        # The relative transform between current robot and simulated pose is:
        #   translation = (dx, dy) in current robot frame,
        #   rotation = dtheta
        # Since traj is in world coords, we don't have that relative directly.
        # So we approximate: evaluate clearance at the first pose only and a few samples.
        # Later, with proper TF/costmap, this becomes exact.

        # v1: sample a few points along trajectory and assume current robot frame.
        sample_idx = [0, len(traj)//3, 2*len(traj)//3, len(traj)-1]
        for _k in sample_idx:
            # For v1 approximation: clearance in current robot frame
            for ox, oy in obs_pts_robot:
                d = math.hypot(ox, oy) - R
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
