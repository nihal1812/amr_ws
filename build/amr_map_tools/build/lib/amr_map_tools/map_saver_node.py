#!/usr/bin/env python3
import os
import time
import yaml
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger


def occ_to_pgm_value(occ: int, negate: bool) -> int:
    """
    Convert ROS occupancy values to PGM grayscale.
    ROS OccupancyGrid data:
      -1 unknown
       0 free
     100 occupied
    Common map convention:
      occupied -> black (0)
      free     -> white (254/255)
      unknown  -> gray (205)
    negate flips black/white if needed.
    """
    if occ == -1:
        v = 205
    else:
        # Map 0..100 to 254..0 (free white, occupied black)
        v = int(round((100 - occ) * 254 / 100))
        if v < 0:
            v = 0
        if v > 254:
            v = 254

    if negate:
        # invert around 254 (keep unknown-ish still invert)
        v = 254 - v
    return v


class MapSaverNode(Node):
    def __init__(self):
        super().__init__("map_saver_node")

        # Parameters
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("output_dir", ".")
        self.declare_parameter("map_name", "my_map")
        self.declare_parameter("negate", False)
        self.declare_parameter("occupied_thresh", 0.65)
        self.declare_parameter("free_thresh", 0.196)

        self._latest_map: Optional[OccupancyGrid] = None
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.VOLATILE
        qos.reliability = ReliabilityPolicy.RELIABLE

        map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self._sub = self.create_subscription(OccupancyGrid, map_topic, self._map_cb, qos)

        self._srv = self.create_service(Trigger, "save_map", self._handle_save_map)

        self.get_logger().info(f"Listening for maps on: {map_topic}")
        self.get_logger().info("Call service to save: ros2 service call /save_map std_srvs/srv/Trigger {}")

    def _map_cb(self, msg: OccupancyGrid):
        self._latest_map = msg
        self.get_logger().info(
            f"Map received: {msg.info.width}x{msg.info.height}, res={msg.info.resolution}"
        )

    def _handle_save_map(self, request, response):
        if self._latest_map is None:
            response.success = False
            response.message = "No map received yet."
            return response

        try:
            out_dir = self.get_parameter("output_dir").get_parameter_value().string_value
            name = self.get_parameter("map_name").get_parameter_value().string_value
            negate = self.get_parameter("negate").get_parameter_value().bool_value
            occupied_thresh = self.get_parameter("occupied_thresh").get_parameter_value().double_value
            free_thresh = self.get_parameter("free_thresh").get_parameter_value().double_value

            os.makedirs(out_dir, exist_ok=True)

            pgm_path = os.path.join(out_dir, f"{name}.pgm")
            yaml_path = os.path.join(out_dir, f"{name}.yaml")

            self._write_pgm(pgm_path, self._latest_map, negate)
            self._write_yaml(yaml_path, pgm_path, self._latest_map, negate, occupied_thresh, free_thresh)

            response.success = True
            response.message = f"Saved map to: {yaml_path} and {pgm_path}"
            self.get_logger().info(response.message)
            return response
        except Exception as e:
            response.success = False
            response.message = f"Failed to save map: {e}"
            self.get_logger().error(response.message)
            return response

    def _write_pgm(self, pgm_path: str, grid: OccupancyGrid, negate: bool):
        w = grid.info.width
        h = grid.info.height
        data = grid.data  # row-major, starts at (0,0) cell in map frame (origin)

        # Write P5 binary PGM
        header = f"P5\n# CREATOR: amr_map_tools {time.time()}\n{w} {h}\n255\n".encode("ascii")

        # Build image bytes: IMPORTANT
        # Convention used by ROS map_server expects the image's (0,0) to correspond to map origin,
        # and the image is stored top-to-bottom. OccupancyGrid is indexed row-major from bottom-left
        # in the map coordinate convention. To match typical map_server expectations, we flip vertically.
        img = bytearray(w * h)
        for y in range(h):
            for x in range(w):
                occ = int(data[y * w + x])
                v = occ_to_pgm_value(occ, negate)
                # flip vertically into image coordinates
                yy = (h - 1 - y)
                img[yy * w + x] = v

        with open(pgm_path, "wb") as f:
            f.write(header)
            f.write(img)

    def _write_yaml(
        self,
        yaml_path: str,
        pgm_path: str,
        grid: OccupancyGrid,
        negate: bool,
        occupied_thresh: float,
        free_thresh: float,
    ):
        # YAML expects image path usually relative; store basename for portability
        image_field = os.path.basename(pgm_path)

        origin = grid.info.origin
        # map_server expects [x, y, yaw]
        # Here we assume yaw=0 if quaternion is identity; for general case we should compute yaw.
        # We'll implement yaw extraction later if you use rotated maps.
        yaw = 0.0

        yml = {
            "image": image_field,
            "mode": "trinary",
            "resolution": float(grid.info.resolution),
            "origin": [float(origin.position.x), float(origin.position.y), float(yaw)],
            "negate": 1 if negate else 0,
            "occupied_thresh": float(occupied_thresh),
            "free_thresh": float(free_thresh),
        }

        with open(yaml_path, "w") as f:
            yaml.safe_dump(yml, f, sort_keys=False)


def main():
    rclpy.init()
    node = MapSaverNode()
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
