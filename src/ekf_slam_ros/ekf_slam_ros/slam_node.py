import numpy as np
import math
import rclpy
from rclpy.node import Node

from ekf_slam.ekf import EKFSLAM

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import LaserScan


# ----------------------------
# Helper functions
# ----------------------------
def yaw_from_quat(q):
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
    y_inv = s * x - c * y
    yaw_inv = -yaw
    return x_inv, y_inv, yaw_inv


def compose_se2(a, b):
    x1, y1, yaw1 = a
    x2, y2, yaw2 = b
    c = math.cos(yaw1)
    s = math.sin(yaw1)
    x = x1 + c * x2 - s * y2
    y = y1 + s * x2 + c * y2
    yaw = math.atan2(math.sin(yaw1 + yaw2), math.cos(yaw1 + yaw2))
    return x, y, yaw


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def pose_delta(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dpos = math.hypot(dx, dy)
    dth = abs(wrap_angle(a[2] - b[2]))
    return dpos, dth


class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        # ----------------------------
        # Parameters
        # ----------------------------
        try:
            self.declare_parameter("use_sim_time", True)
        except Exception:
            pass

        # EKF motion noise
        self.declare_parameter("ekf.sigma_v", 0.05)
        self.declare_parameter("ekf.sigma_w", 0.20)

        # Scan matching parameters
        self.declare_parameter("scan_match.enable", False)
        self.declare_parameter("scan_match.beam_step", 8)
        self.declare_parameter("scan_match.occ_thresh", 1.0)
        self.declare_parameter("scan_match.range_margin", 0.10)
        self.declare_parameter("scan_match.min_occupied_cells_to_enable", 150)
        self.declare_parameter("scan_match.min_score_to_accept", 12)
        self.declare_parameter("scan_match.max_translation_jump", 0.08)
        self.declare_parameter("scan_match.max_rotation_jump_deg", 8.0)
        self.declare_parameter("scan_match.skip_map_update_if_rejected", True)

        # Coarse-to-fine scan matching search parameters
        self.declare_parameter("scan_match.coarse_dx", 0.12)
        self.declare_parameter("scan_match.coarse_dy", 0.12)
        self.declare_parameter("scan_match.coarse_dth_deg", 6.0)
        self.declare_parameter("scan_match.coarse_step_xy", 0.03)
        self.declare_parameter("scan_match.coarse_step_th_deg", 2.0)

        self.declare_parameter("scan_match.fine_dx", 0.04)
        self.declare_parameter("scan_match.fine_dy", 0.04)
        self.declare_parameter("scan_match.fine_dth_deg", 2.0)
        self.declare_parameter("scan_match.fine_step_xy", 0.01)
        self.declare_parameter("scan_match.fine_step_th_deg", 0.5)

        # SLAM / debug
        self.declare_parameter("slam.fast_rotation_threshold", 1.0)
        self.declare_parameter("slam.update_period", 0.2)
        self.declare_parameter("slam.skip_map_update_during_fast_rotation", True)
        self.declare_parameter("slam.debug_print_period", 1.0)
        self.declare_parameter("slam.max_scan_odom_dt", 0.10)

        # Map parameters
        self.declare_parameter("map.resolution", 0.05)
        self.declare_parameter("map.width", 800)
        self.declare_parameter("map.height", 800)
        self.declare_parameter("map.origin_x", -20.0)
        self.declare_parameter("map.origin_y", -20.0)

        # Mapping parameters
        self.declare_parameter("mapping.l_hit", 0.65)
        self.declare_parameter("mapping.l_miss", 0.25)
        self.declare_parameter("mapping.l_min", -5.0)
        self.declare_parameter("mapping.l_max", 5.0)
        self.declare_parameter("mapping.map_beam_step", 4)
        self.declare_parameter("mapping.min_valid_range", 0.08)

        # ----------------------------
        # Read params
        # ----------------------------
        sigma_v = float(self.get_parameter("ekf.sigma_v").value)
        sigma_w = float(self.get_parameter("ekf.sigma_w").value)
        self.Q = np.diag([sigma_v ** 2, sigma_w ** 2])

        self.update_period = float(self.get_parameter("slam.update_period").value)
        self.max_scan_odom_dt = float(self.get_parameter("slam.max_scan_odom_dt").value)

        self.res = float(self.get_parameter("map.resolution").value)
        self.width = int(self.get_parameter("map.width").value)
        self.height = int(self.get_parameter("map.height").value)
        self.origin_x = float(self.get_parameter("map.origin_x").value)
        self.origin_y = float(self.get_parameter("map.origin_y").value)

        self.l_hit = float(self.get_parameter("mapping.l_hit").value)
        self.l_miss = float(self.get_parameter("mapping.l_miss").value)
        self.l_min = float(self.get_parameter("mapping.l_min").value)
        self.l_max = float(self.get_parameter("mapping.l_max").value)
        self.map_beam_step = int(self.get_parameter("mapping.map_beam_step").value)
        self.min_valid_range = float(self.get_parameter("mapping.min_valid_range").value)

        self.enable_scan_matching = bool(self.get_parameter("scan_match.enable").value)
        self.sm_beam_step = int(self.get_parameter("scan_match.beam_step").value)
        self.occ_thresh = float(self.get_parameter("scan_match.occ_thresh").value)
        self.range_margin = float(self.get_parameter("scan_match.range_margin").value)
        self.min_occupied_cells_to_enable = int(self.get_parameter("scan_match.min_occupied_cells_to_enable").value)
        self.min_score_to_accept = int(self.get_parameter("scan_match.min_score_to_accept").value)

        self.max_correction_xy = float(
            self.get_parameter("scan_match.max_translation_jump").value
        )
        self.max_correction_th = math.radians(
            float(self.get_parameter("scan_match.max_rotation_jump_deg").value)
        )
        self.skip_map_update_on_reject = bool(
            self.get_parameter("scan_match.skip_map_update_if_rejected").value
        )

        self.coarse_dx = float(self.get_parameter("scan_match.coarse_dx").value)
        self.coarse_dy = float(self.get_parameter("scan_match.coarse_dy").value)
        self.coarse_dth = math.radians(float(self.get_parameter("scan_match.coarse_dth_deg").value))
        self.coarse_step_xy = float(self.get_parameter("scan_match.coarse_step_xy").value)
        self.coarse_step_th = math.radians(float(self.get_parameter("scan_match.coarse_step_th_deg").value))

        self.fine_dx = float(self.get_parameter("scan_match.fine_dx").value)
        self.fine_dy = float(self.get_parameter("scan_match.fine_dy").value)
        self.fine_dth = math.radians(float(self.get_parameter("scan_match.fine_dth_deg").value))
        self.fine_step_xy = float(self.get_parameter("scan_match.fine_step_xy").value)
        self.fine_step_th = math.radians(float(self.get_parameter("scan_match.fine_step_th_deg").value))

        self.skip_map_update_during_fast_rotation = bool(
            self.get_parameter("slam.skip_map_update_during_fast_rotation").value
        )
        self.fast_rotation_threshold = float(self.get_parameter("slam.fast_rotation_threshold").value)
        self.debug_print_period = float(self.get_parameter("slam.debug_print_period").value)

        # ----------------------------
        # Publishers / TF
        # ----------------------------
        self.pose_pub = self.create_publisher(PoseStamped, '/slam_pose', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ----------------------------
        # EKF state
        # ----------------------------
        init_pose = np.array([0.0, 0.0, 0.0])
        init_cov = np.eye(3) * 1e-3
        self.ekf = EKFSLAM(init_pose, init_cov)
        self.last_stamp = None

        # Latest state
        self.odom = None
        self.last_scan = None
        self.last_scan_stamp = None
        self.T_odom_base_latest = (0.0, 0.0, 0.0)
        self.T_map_base_pred = None
        self.last_pred_stamp = None
        self.last_good_map_base = None

        # Map storage
        self.logodds = np.zeros((self.height, self.width), dtype=np.float32)

        # Debug
        self.last_debug_print_time = 0.0

        # ----------------------------
        # Subscriptions
        # ----------------------------
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 50)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        # Slow loop
        self.timer = self.create_timer(self.update_period, self.slam_update_cb)

        self.get_logger().info("slam_node started: EKF predict + OccupancyGrid + gated scan matching")

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_stamp = msg.header.stamp

    def odom_cb(self, msg: Odometry):
        self.odom = msg

        x_o = msg.pose.pose.position.x
        y_o = msg.pose.pose.position.y
        yaw_o = yaw_from_quat(msg.pose.pose.orientation)
        self.T_odom_base_latest = (float(x_o), float(y_o), float(yaw_o))

        t_now = stamp_to_seconds(msg.header.stamp)
        if self.last_stamp is None:
            self.ekf.mu[0:3] = np.array([x_o, y_o, yaw_o])
            self.last_stamp = t_now
            self.T_map_base_pred = (float(x_o), float(y_o), float(yaw_o))
            self.last_pred_stamp = msg.header.stamp
            self.last_good_map_base = self.T_map_base_pred
            return

        dt = t_now - self.last_stamp
        self.last_stamp = t_now
        if dt <= 0.0 or dt > 0.5:
            return

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

        self.ekf.predict((v, w, dt), self.Q)
        x_s, y_s, yaw_s = self.ekf.pose()

        self.T_map_base_pred = (float(x_s), float(y_s), float(yaw_s))
        self.last_pred_stamp = msg.header.stamp

    def slam_update_cb(self):
        if self.last_scan is None or self.T_map_base_pred is None or self.last_pred_stamp is None:
            return

        if self.last_scan_stamp is None:
            return

        scan_t = stamp_to_seconds(self.last_scan_stamp)
        pred_t = stamp_to_seconds(self.last_pred_stamp)

        if abs(scan_t - pred_t) > self.max_scan_odom_dt:
            return

        scan = self.last_scan
        T_pred = self.T_map_base_pred
        stamp = self.last_scan_stamp
        now_s = self.get_clock().now().nanoseconds / 1e9

        occ_cells = int(np.count_nonzero(self.logodds > self.occ_thresh))
        can_match = (occ_cells >= self.min_occupied_cells_to_enable)

        T_map_base = T_pred
        accepted_match = False
        rejected_due_to_jump = False
        scan_match_score = -1

        if self.enable_scan_matching and can_match:
            T_candidate, scan_match_score = self.scan_match(scan, T_pred)
            dpos, dth = pose_delta(T_candidate, T_pred)

            if (
                scan_match_score >= self.min_score_to_accept
                and dpos <= self.max_correction_xy
                and dth <= self.max_correction_th
            ):
                T_map_base = T_candidate
                accepted_match = True
            else:
                if scan_match_score >= self.min_score_to_accept:
                    rejected_due_to_jump = True

        do_map_update = True
        map_update_reason = "normal"

        if rejected_due_to_jump and self.skip_map_update_on_reject:
            do_map_update = False
            map_update_reason = "reject_jump"

        current_w = 0.0
        if self.odom is not None:
            current_w = float(self.odom.twist.twist.angular.z)

        if self.skip_map_update_during_fast_rotation and abs(current_w) > self.fast_rotation_threshold:
            do_map_update = False
            map_update_reason = "fast_rotation"

        if do_map_update:
            self.integrate_scan(scan, T_map_base)

        if accepted_match:
            self.ekf.mu[0:3] = np.array(
                [T_map_base[0], T_map_base[1], T_map_base[2]],
                dtype=float
            )
            self.last_good_map_base = T_map_base
        elif self.last_good_map_base is None:
            self.last_good_map_base = T_map_base

        T_base_odom = invert_se2(*self.T_odom_base_latest)
        T_map_odom = compose_se2(T_map_base, T_base_odom)
        self.broadcast_map_to_odom(stamp, T_map_odom)

        self.publish_slam_pose(stamp, T_map_base)
        self.publish_map(stamp)

        if now_s - self.last_debug_print_time >= self.debug_print_period:
            self.last_debug_print_time = now_s
            msg = (
                f"[SLAM] occ_cells={occ_cells} "
                f"scan_match_score={scan_match_score} "
                f"accepted={accepted_match} "
                f"map_update={do_map_update}({map_update_reason}) "
                f"pred=({T_pred[0]:.2f},{T_pred[1]:.2f},{math.degrees(T_pred[2]):.1f}deg) "
                f"used=({T_map_base[0]:.2f},{T_map_base[1]:.2f},{math.degrees(T_map_base[2]):.1f}deg) "
                f"w={current_w:.2f} dt={abs(scan_t - pred_t):.3f}"
            )
            self.get_logger().info(msg)

    # ----------------------------
    # Map utilities
    # ----------------------------
    def world_to_grid(self, x, y):
        i = int((x - self.origin_x) / self.res)
        j = int((y - self.origin_y) / self.res)
        if i < 0 or i >= self.width or j < 0 or j >= self.height:
            return None
        return (i, j)

    def in_bounds(self, i, j):
        return (0 <= i < self.width) and (0 <= j < self.height)

    def bresenham(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    def integrate_scan(self, scan: LaserScan, T_map_base):
        x, y, yaw = T_map_base
        start = self.world_to_grid(x, y)
        if start is None:
            return
        sx, sy = start

        angle = scan.angle_min
        idx = 0
        max_hit_range = scan.range_max - self.range_margin

        for r in scan.ranges:
            if idx % self.map_beam_step != 0:
                angle += scan.angle_increment
                idx += 1
                continue

            if (not math.isfinite(r)) or r < max(scan.range_min, self.min_valid_range) or r > scan.range_max:
                angle += scan.angle_increment
                idx += 1
                continue

            ex = x + r * math.cos(yaw + angle)
            ey = y + r * math.sin(yaw + angle)

            end = self.world_to_grid(ex, ey)
            if end is None:
                angle += scan.angle_increment
                idx += 1
                continue
            exi, eyi = end

            line = self.bresenham(sx, sy, exi, eyi)
            if len(line) < 2:
                angle += scan.angle_increment
                idx += 1
                continue

            for (cx, cy) in line[:-1]:
                if self.in_bounds(cx, cy):
                    self.logodds[cy, cx] = max(self.l_min, self.logodds[cy, cx] - self.l_miss)

            if r < max_hit_range and self.in_bounds(exi, eyi):
                self.logodds[eyi, exi] = min(self.l_max, self.logodds[eyi, exi] + self.l_hit)

            angle += scan.angle_increment
            idx += 1

    def publish_map(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"

        msg.info.resolution = float(self.res)
        msg.info.width = int(self.width)
        msg.info.height = int(self.height)
        msg.info.origin.position.x = float(self.origin_x)
        msg.info.origin.position.y = float(self.origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        p = 1.0 / (1.0 + np.exp(-self.logodds))
        occ = (p * 100.0).astype(np.int8)

        unknown = np.abs(self.logodds) < 0.1
        occ[unknown] = -1

        msg.data = [int(v) for v in occ.flatten(order='C')]
        self.map_pub.publish(msg)

    # ----------------------------
    # Scan matching
    # ----------------------------
    def score_pose(self, scan: LaserScan, T_map_base):
        x, y, yaw = T_map_base
        score = 0

        angle = scan.angle_min
        idx = 0
        max_hit_range = scan.range_max - self.range_margin

        for r in scan.ranges:
            if idx % self.sm_beam_step != 0:
                angle += scan.angle_increment
                idx += 1
                continue

            if (not math.isfinite(r)) or r < max(scan.range_min, self.min_valid_range) or r > scan.range_max:
                angle += scan.angle_increment
                idx += 1
                continue

            if r >= max_hit_range:
                angle += scan.angle_increment
                idx += 1
                continue

            ex = x + r * math.cos(yaw + angle)
            ey = y + r * math.sin(yaw + angle)

            g = self.world_to_grid(ex, ey)
            if g is not None:
                i, j = g
                if self.logodds[j, i] > self.occ_thresh:
                    score += 1

            angle += scan.angle_increment
            idx += 1

        return score

    def scan_match_window(self, scan, center_pose, dx, dy, dth, step_xy, step_th):
        x0, y0, th0 = center_pose

        dxs = np.arange(-dx, dx + 1e-9, step_xy)
        dys = np.arange(-dy, dy + 1e-9, step_xy)
        dts = np.arange(-dth, dth + 1e-12, step_th)

        best_pose = center_pose
        best_score = -1e9

        for dtheta in dts:
            th = wrap_angle(th0 + float(dtheta))
            for ddx in dxs:
                for ddy in dys:
                    pose = (x0 + float(ddx), y0 + float(ddy), th)
                    s = self.score_pose(scan, pose)
                    if s > best_score:
                        best_score = s
                        best_pose = pose

        return best_pose, best_score

    def scan_match(self, scan: LaserScan, T_pred):
        coarse_best, _ = self.scan_match_window(
            scan,
            T_pred,
            dx=self.coarse_dx,
            dy=self.coarse_dy,
            dth=self.coarse_dth,
            step_xy=self.coarse_step_xy,
            step_th=self.coarse_step_th
        )
        fine_best, fine_score = self.scan_match_window(
            scan,
            coarse_best,
            dx=self.fine_dx,
            dy=self.fine_dy,
            dth=self.fine_dth,
            step_xy=self.fine_step_xy,
            step_th=self.fine_step_th
        )
        return fine_best, fine_score

    # ----------------------------
    # Publishing helpers
    # ----------------------------
    def broadcast_map_to_odom(self, stamp, T_map_odom):
        x_mo, y_mo, yaw_mo = T_map_odom
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = float(x_mo)
        t.transform.translation.y = float(y_mo)
        t.transform.translation.z = 0.0
        t.transform.rotation = quat_from_yaw(float(yaw_mo))
        self.tf_broadcaster.sendTransform(t)

    def publish_slam_pose(self, stamp, T_map_base):
        x, y, yaw = T_map_base
        out = PoseStamped()
        out.header.stamp = stamp
        out.header.frame_id = "map"
        out.pose.position.x = float(x)
        out.pose.position.y = float(y)
        out.pose.position.z = 0.0
        out.pose.orientation = quat_from_yaw(float(yaw))
        self.pose_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
