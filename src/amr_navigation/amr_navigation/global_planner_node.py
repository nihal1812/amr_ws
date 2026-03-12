#!/usr/bin/env python3
from __future__ import annotations

import math
import heapq
from typing import Optional, List, Tuple, Dict

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

from .grid_utils import world_to_grid, grid_to_world


Cell = Tuple[int, int]


class GlobalPlanner(Node):
    def __init__(self):
        super().__init__("amr_global_planner")

        # Topics
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("slam_pose_topic", "/slam_pose")
        self.declare_parameter("goal_topic", "/goal_pose")

        # Occupancy handling
        self.declare_parameter("occ_threshold", 50)
        self.declare_parameter("treat_unknown_as_free", True)

        # Planning
        self.declare_parameter("plan_period", 0.5)
        self.declare_parameter("max_expansions", 300000)

        # Safety inflation in cells
        self.declare_parameter("inflate_cells", 1)

        self.map_msg: Optional[OccupancyGrid] = None
        self.slam_pose: Optional[PoseStamped] = None
        self.goal_pose: Optional[PoseStamped] = None

        self._last_path_len = 0
        self._last_warn = ""

        map_topic = self.get_parameter("map_topic").value
        path_topic = self.get_parameter("path_topic").value
        slam_pose_topic = self.get_parameter("slam_pose_topic").value
        goal_topic = self.get_parameter("goal_topic").value
        plan_period = float(self.get_parameter("plan_period").value)

        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, 10)
        self.create_subscription(PoseStamped, slam_pose_topic, self._slam_pose_cb, 20)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 10)

        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.timer = self.create_timer(plan_period, self._tick)

        self.get_logger().info(f"Subscribed map: {map_topic}")
        self.get_logger().info(f"Subscribed slam pose: {slam_pose_topic}")
        self.get_logger().info(f"Subscribed goal: {goal_topic}")
        self.get_logger().info(f"Publishing path: {path_topic}")

    # ---------------- ROS callbacks ----------------
    def _map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg

    def _slam_pose_cb(self, msg: PoseStamped):
        self.slam_pose = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal_pose = msg
        self.get_logger().info(
            f"Received new goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) in frame '{msg.header.frame_id}'"
        )

    def _tick(self):
        if self.map_msg is None or self.slam_pose is None or self.goal_pose is None:
            return

        if self.goal_pose.header.frame_id and self.goal_pose.header.frame_id != "map":
            if self._last_warn != "goal_not_map":
                self.get_logger().warn(
                    f"Goal frame is '{self.goal_pose.header.frame_id}', expected 'map'. Ignoring goal."
                )
                self._last_warn = "goal_not_map"
            return

        sx = float(self.slam_pose.pose.position.x)
        sy = float(self.slam_pose.pose.position.y)
        gx = float(self.goal_pose.pose.position.x)
        gy = float(self.goal_pose.pose.position.y)

        cells = self.plan_astar_world(sx, sy, gx, gy)
        if not cells:
            if self._last_warn != "no_path":
                self.get_logger().warn("No path found (or start/goal not free).")
                self._last_warn = "no_path"
            return

        self._last_warn = ""
        msg = self.cells_to_path(cells)
        self.path_pub.publish(msg)

        if len(cells) != self._last_path_len:
            self.get_logger().info(f"Published path with {len(cells)} poses.")
            self._last_path_len = len(cells)

    # ---------------- Planning ----------------
    def plan_astar_world(self, sx: float, sy: float, gx: float, gy: float) -> List[Cell]:
        m = self.map_msg
        assert m is not None

        res = float(m.info.resolution)
        ox = float(m.info.origin.position.x)
        oy = float(m.info.origin.position.y)
        w = int(m.info.width)
        h = int(m.info.height)

        start = world_to_grid(sx, sy, ox, oy, res)
        goal = world_to_grid(gx, gy, ox, oy, res)

        if start is None or goal is None:
            if self._last_warn != "start_or_goal_oob":
                self.get_logger().warn("Start or goal is outside map bounds.")
                self._last_warn = "start_or_goal_oob"
            return []

        data = self.inflate_map_if_needed(w, h, m.data)
        return self.astar(start, goal, w, h, data)

    def inflate_map_if_needed(self, w: int, h: int, data) -> List[int]:
        inflate_cells = int(self.get_parameter("inflate_cells").value)
        occ_threshold = int(self.get_parameter("occ_threshold").value)
        treat_unknown_as_free = bool(self.get_parameter("treat_unknown_as_free").value)

        src = list(data)
        if inflate_cells <= 0:
            return src

        inflated = src[:]
        occupied = []

        for j in range(h):
            row = j * w
            for i in range(w):
                v = int(src[row + i])
                if v >= occ_threshold or (v < 0 and not treat_unknown_as_free):
                    occupied.append((i, j))

        r2 = inflate_cells * inflate_cells
        for (cx, cy) in occupied:
            for dj in range(-inflate_cells, inflate_cells + 1):
                for di in range(-inflate_cells, inflate_cells + 1):
                    if di * di + dj * dj > r2:
                        continue
                    ni = cx + di
                    nj = cy + dj
                    if 0 <= ni < w and 0 <= nj < h:
                        inflated[nj * w + ni] = 100

        return inflated

    def nearest_free_cell(self, cell: Cell, w: int, h: int, data, max_radius: int = 8) -> Optional[Cell]:
        ci, cj = cell

        if self.is_free(ci, cj, w, h, data):
            return cell

        for r in range(1, max_radius + 1):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    ni = ci + di
                    nj = cj + dj

                    if abs(di) != r and abs(dj) != r:
                        continue

                    if self.is_free(ni, nj, w, h, data):
                        return (ni, nj)

        return None

    def astar(self, start: Cell, goal: Cell, w: int, h: int, data) -> List[Cell]:
        start_free = self.nearest_free_cell(start, w, h, data, max_radius=8)
        if start_free is None:
            if self._last_warn != "start_blocked":
                self.get_logger().warn("Start cell is not free, and no nearby free cell was found.")
                self._last_warn = "start_blocked"
            return []
        if start_free != start:
            self.get_logger().warn(
                f"Start cell blocked. Using nearby free cell {start_free} instead of {start}."
            )
        start = start_free

        goal_free = self.nearest_free_cell(goal, w, h, data, max_radius=20)
        if goal_free is None:
            if self._last_warn != "goal_blocked":
                self.get_logger().warn(
                    f"Goal cell {goal} is not free, and no nearby free cell was found within radius 20."
                )
                self._last_warn = "goal_blocked"
            return []
        if goal_free != goal:
            self.get_logger().warn(
                f"Goal cell blocked. Using nearby free cell {goal_free} instead of {goal}."
            )
        goal = goal_free

        if start == goal:
            self.get_logger().warn(
                f"Start and goal collapse to the same free cell {start}. Goal is too close or blocked."
            )
            return [start]

        max_expansions = int(self.get_parameter("max_expansions").value)

        open_heap: List[Tuple[float, float, Cell]] = []
        came_from: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, float] = {start: 0.0}
        closed = set()

        h0 = self.heuristic_octile(start, goal)
        heapq.heappush(open_heap, (h0, h0, start))

        expansions = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            expansions += 1
            if expansions > max_expansions:
                if self._last_warn != "expansion_limit":
                    self.get_logger().warn("A*: expansion limit reached, aborting.")
                    self._last_warn = "expansion_limit"
                return []

            cx, cy = current
            current_g = g_score[current]

            for nx, ny in self.neighbors8_no_corner_cut(cx, cy, w, h, data):
                if not self.is_free(nx, ny, w, h, data):
                    continue

                neighbor: Cell = (nx, ny)
                if neighbor in closed:
                    continue

                tentative_g = current_g + self.step_cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    hval = self.heuristic_octile(neighbor, goal)
                    fval = tentative_g + hval
                    heapq.heappush(open_heap, (fval, hval, neighbor))

        return []

    # ---------------- Grid helpers ----------------
    def is_free(self, i: int, j: int, w: int, h: int, data) -> bool:
        if i < 0 or j < 0 or i >= w or j >= h:
            return False

        v = int(data[j * w + i])

        if v < 0:
            return bool(self.get_parameter("treat_unknown_as_free").value)

        return v < int(self.get_parameter("occ_threshold").value)

    def neighbors8_no_corner_cut(self, i: int, j: int, w: int, h: int, data):
        dirs = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        for di, dj in dirs:
            ni, nj = i + di, j + dj

            if di != 0 and dj != 0:
                if not self.is_free(i + di, j, w, h, data):
                    continue
                if not self.is_free(i, j + dj, w, h, data):
                    continue

            yield ni, nj

    def heuristic_octile(self, a: Cell, b: Cell) -> float:
        ax, ay = a
        bx, by = b
        dx = abs(ax - bx)
        dy = abs(ay - by)
        return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)

    def step_cost(self, a: Cell, b: Cell) -> float:
        ax, ay = a
        bx, by = b
        return math.sqrt(2.0) if (ax != bx and ay != by) else 1.0

    def reconstruct_path(self, came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ---------------- Publishing helpers ----------------
    def cells_to_path(self, cells: List[Cell]) -> Path:
        m = self.map_msg
        assert m is not None

        res = float(m.info.resolution)
        ox = float(m.info.origin.position.x)
        oy = float(m.info.origin.position.y)

        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for (i, j) in cells:
            x, y = grid_to_world(i, j, ox, oy, res)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        return path


def main():
    rclpy.init()
    node = GlobalPlanner()
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
