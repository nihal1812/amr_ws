#!/usr/bin/env python3
import os
import math
import yaml
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose


def read_pgm_p5(path: str) -> Tuple[int, int, bytes]:
    """
    Minimal reader for binary PGM (P5).
    Returns width, height, raw pixel bytes (len = w*h).
    """
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Unsupported PGM magic {magic}, expected P5")

        # Skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        # Read width height
        parts = line.split()
        if len(parts) != 2:
            parts = (line + f.readline()).split()
        w, h = int(parts[0]), int(parts[1])

        maxval = int(f.readline().strip())
        if maxval > 255:
            raise ValueError("Only 8-bit PGM supported (maxval <= 255)")

        data = f.read(w * h)
        if len(data) != w * h:
            raise ValueError("PGM data size mismatch")
        return w, h, data


def pgm_value_to_occ(v: int, negate: bool, occupied_thresh: float, free_thresh: float) -> int:
    """
    Convert PGM pixel value back to OccupancyGrid cell:
      free -> 0
      occupied -> 100
      unknown -> -1
    Using thresholds on normalized occupancy.
    """
    if negate:
        v = 254 - v

    # Convert pixel value to occupancy probability [0..1]
    # Following same scale used in saver: v ~ (100-occ)*254/100
    # Normalize: occ_prob ~ 1 - (v/254)
    occ_prob = 1.0 - (float(v) / 254.0)

    if occ_prob > occupied_thresh:
        return 100
    if occ_prob < free_thresh:
        return 0
    return -1


class MapLoaderNode(Node):
    def __init__(self):
        super().__init__("map_loader_node")

        self.declare_parameter("yaml_path", "")
        self.declare_parameter("map_topic", "/map")

        yaml_path = self.get_parameter("yaml_path").get_parameter_value().string_value
        if not yaml_path:
            raise RuntimeError("Parameter yaml_path is required, e.g. --ros-args -p yaml_path:=/path/to/my_map.yaml")

        map_topic = self.get_parameter("map_topic").get_parameter_value().string_value

        # Latching-like behavior in ROS2: transient local durability
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE

        self._pub = self.create_publisher(OccupancyGrid, map_topic, qos)
        if not os.path.exists(yaml_path):
            self.get_logger().error(f"YAML file not found: {yaml_path}")
            self.get_logger().error("Create it by saving a map or point yaml_path to an existing map file.")
            # Keep node alive so you can see the error and fix parameters without stack traces
            self._timer = self.create_timer(1.0, lambda: None)
            return

        grid = self._load_map(yaml_path)
        self._pub.publish(grid)
        self.get_logger().info(f"Published map on {map_topic} from {yaml_path}")

    def _load_map(self, yaml_path: str) -> OccupancyGrid:
        base_dir = os.path.dirname(os.path.abspath(yaml_path))
        with open(yaml_path, "r") as f:
            yml = yaml.safe_load(f)

        image = yml["image"]
        image_path = image if os.path.isabs(image) else os.path.join(base_dir, image)

        resolution = float(yml["resolution"])
        origin = yml["origin"]  # [x,y,yaw]
        negate = bool(int(yml.get("negate", 0)))
        occupied_thresh = float(yml.get("occupied_thresh", 0.65))
        free_thresh = float(yml.get("free_thresh", 0.196))

        w, h, pixels = read_pgm_p5(image_path)

        msg = OccupancyGrid()
        msg.header.frame_id = "map"  # standard
        msg.info.resolution = resolution
        msg.info.width = w
        msg.info.height = h

        # origin pose
        msg.info.origin.position.x = float(origin[0])
        msg.info.origin.position.y = float(origin[1])
        yaw = float(origin[2]) if len(origin) > 2 else 0.0
        # yaw -> quaternion (z-w only)
        msg.info.origin.orientation.z = math.sin(yaw / 2.0)
        msg.info.origin.orientation.w = math.cos(yaw / 2.0)

        data = [0] * (w * h)

        # Inverse of our saver flip: PGM stored top-to-bottom, OccupancyGrid expects bottom-to-top indexing.
        for y in range(h):
            for x in range(w):
                yy = (h - 1 - y)
                v = pixels[y * w + x]
                occ = pgm_value_to_occ(v, negate, occupied_thresh, free_thresh)
                data[yy * w + x] = occ

        msg.data = data
        return msg


def main():
    rclpy.init()
    node = MapLoaderNode()
    try:
        rclpy.spin(node)  # keeps node alive for late subscribers
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
