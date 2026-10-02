#!/usr/bin/env python3
"""
ROS 2 node: the mine map. Fuses the geo-located detections of all scanning drones
(mine_detector_node.py) into one list of mines in the arena frame, keeps the
coverage / occupancy grid with the 1 m exclusion zones, and saves everything.
Runs next to the mother's localizer: she owns the arena frame and the map.

Subscribes  /NAME/mines/detections (String JSON) for NAME in `drones`
            /NAME/mavros/local_position/pose for NAME in drones + `others`
            (drawn on the map; detections on another drone, landed or flying below the
            camera, are ignored)
Publishes   /mines/map        std_msgs/String (JSON, transient local)
            /mines/markers    visualization_msgs/MarkerArray (frame "arena")
            /mines/grid       nav_msgs/OccupancyGrid (frame "arena", transient local)
            /mines/map_image  sensor_msgs/Image (top-down picture)
Saves       out_dir/mines.json, mines.csv, mine_map.png, grid.pgm + grid.yaml (map_server)

    python3 mine_map_node.py --ros-args -p drones:="['DeeDee','Marky']" -p out_dir:=mine_map
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mine_map import MineMap                                        # noqa: E402

COLOURS = {"mine_disc": (0.9, 0.15, 0.15), "surface_marker": (1.0, 0.6, 0.0)}


class MineMapNode(Node):
    def __init__(self):
        super().__init__("mine_map")
        p = self.declare_parameter
        self.drones = list(p("drones", ["DeeDee", "Marky"]).value)
        self.others = list(p("others", ["Joey"]).value)
        self.frame_id = p("frame_id", "arena").value
        self.out_dir = os.path.abspath(os.path.expanduser(p("out_dir", "mine_map").value))
        self.clearance = p("drone_clearance", 0.45).value         # a drone's radius on the ground (m)
        self.period = p("publish_period", 1.0).value
        truth_file = p("truth_file", "").value                    # sim only: ground truth to draw / score
        field_tf = list(p("field_tf", [0.0]).value)               # [x0, y0, yaw_deg]: arena -> field frame
        self.map = MineMap(
            assoc_radius=p("assoc_radius", 0.5).value, min_hits=p("min_hits", 4).value,
            min_ratio=p("min_ratio", 0.25).value, max_spread=p("max_spread", 0.2).value,
            exclusion_radius=p("exclusion_radius", 1.0).value,
            mine_radius=p("mine_radius", 0.15).value, grid_resolution=p("grid_resolution", 0.1).value,
            field_tf=(field_tf[0], field_tf[1], math.radians(field_tf[2])) if len(field_tf) == 3 else None)
        self.truth = []
        if truth_file:
            t = json.load(open(truth_file))
            self.truth = [(m["arena_x"], m["arena_y"]) for m in t.get("mines", t)]
        os.makedirs(self.out_dir, exist_ok=True)

        self.positions = {}
        self.stats = {d: 0 for d in self.drones}
        self.ignored = 0
        self.saved_version = -1
        self.last_save = 0.0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(String, "/mines/map", latched)
        self.grid_pub = self.create_publisher(OccupancyGrid, "/mines/grid", latched)
        self.marker_pub = self.create_publisher(MarkerArray, "/mines/markers", 2)
        self.image_pub = self.create_publisher(Image, "/mines/map_image", 2)
        for d in self.drones:
            self.create_subscription(String, f"/{d}/mines/detections", lambda m, d=d: self.on_detections(d, m), 20)
        for d in self.drones + self.others:
            self.create_subscription(PoseStamped, f"/{d}/mavros/local_position/pose",
                                     lambda m, d=d: self.on_pose(d, m), qos_profile_sensor_data)
        self.create_timer(self.period, self.publish)
        self.get_logger().info(f"mine map from {self.drones}, saving to {self.out_dir}")

    def on_pose(self, name, msg):
        p = msg.pose.position
        self.positions[name] = (p.x, p.y, p.z)

    def drone_shadows(self, drone, cam):
        """Where every other drone below this camera appears on the ground plane: the ray from
        the camera through the drone, extended to the ground. A landed drone sits at its own
        position; the mother looking at a daughter flying at 3 m sees her ~2x further out, and
        bigger by the same factor. -> [(x, y, radius)]"""
        out = []
        for n, (x, y, z) in self.positions.items():
            if n == drone or z >= cam[2] - 0.3:          # itself, or not below this camera
                continue
            k = cam[2] / max(cam[2] - z, 0.1)
            out.append((cam[0] + (x - cam[0]) * k, cam[1] + (y - cam[1]) * k, self.clearance * k))
        return out

    def on_detections(self, drone, msg):
        f = json.loads(msg.data)
        pose = f.get("pose")
        cam = (pose[0], pose[1], pose[2] - 0.125) if pose else None
        shadows = self.drone_shadows(drone, cam) if cam else []
        dets = []
        for d in f["detections"]:
            if d["x"] is None:
                continue
            if any(math.hypot(d["x"] - x, d["y"] - y) < r for x, y, r in shadows):
                self.ignored += 1                      # it is (part of) another drone
                continue
            dets.append(d)
        self.map.add_frame(dets, np.array(f["footprint"]) if f["footprint"] else None, t=f["stamp"], source=drone)
        self.stats[drone] += 1

    # ----------------------------------------------------------------- outputs
    def publish(self):
        now = self.get_clock().now().to_msg()
        data = self.map.to_dict()
        self.map_pub.publish(String(data=json.dumps(data)))
        self.publish_markers(now)
        self.publish_grid(now)
        img = self.map.render(drones=self.positions, truth=self.truth)
        m = Image()
        m.header.stamp, m.header.frame_id = now, self.frame_id
        m.height, m.width = img.shape[:2]
        m.encoding, m.step = "bgr8", img.shape[1] * 3
        m.data = img.tobytes()
        self.image_pub.publish(m)
        if self.map.version != self.saved_version or time.time() - self.last_save > 10.0:
            self.save(img)

    def publish_markers(self, stamp):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, mine in enumerate(self.map.mines(True)):
            r, g, b = COLOURS.get(mine["class"], (0.8, 0.8, 0.8))
            for k, (radius, alpha, h) in enumerate(((self.map.mine_radius, 1.0, 0.08),
                                                    (self.map.mine_radius + self.map.exclusion_radius, 0.25, 0.01))):
                mk = Marker()
                mk.header.stamp, mk.header.frame_id = stamp, self.frame_id
                mk.ns, mk.id, mk.type, mk.action = ("mines", "exclusion")[k], mine["id"], Marker.CYLINDER, Marker.ADD
                mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = mine["x"], mine["y"], h / 2
                mk.pose.orientation.w = 1.0
                mk.scale.x = mk.scale.y = 2 * radius
                mk.scale.z = h
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, alpha
                arr.markers.append(mk)
            t = Marker()
            t.header.stamp, t.header.frame_id = stamp, self.frame_id
            t.ns, t.id, t.type, t.action = "labels", mine["id"], Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = mine["x"], mine["y"], 0.4
            t.pose.orientation.w = 1.0
            t.scale.z = 0.25
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"#{mine['id']} ({mine['x']:.2f}, {mine['y']:.2f})"
            arr.markers.append(t)
        self.marker_pub.publish(arr)

    def publish_grid(self, stamp):
        occ, x0, y0, res = self.map.occupancy()
        g = OccupancyGrid()
        g.header.stamp, g.header.frame_id = stamp, self.frame_id
        g.info.resolution = float(res)
        g.info.height, g.info.width = occ.shape
        g.info.origin.position.x, g.info.origin.position.y = float(x0), float(y0)
        g.info.origin.orientation.w = 1.0
        g.data = occ.ravel().tolist()
        self.grid_pub.publish(g)

    def save(self, img):
        self.map.save_json(os.path.join(self.out_dir, "mines.json"))
        self.map.save_csv(os.path.join(self.out_dir, "mines.csv"))
        cv2.imwrite(os.path.join(self.out_dir, "mine_map.png"), img)
        occ, x0, y0, res = self.map.occupancy()
        pgm = np.where(occ < 0, 205, 254 - (occ.astype(np.int16) * 254 // 100)).astype(np.uint8)[::-1]
        cv2.imwrite(os.path.join(self.out_dir, "grid.pgm"), pgm)
        with open(os.path.join(self.out_dir, "grid.yaml"), "w") as f:
            f.write(f"image: grid.pgm\nmode: trinary\nresolution: {res}\norigin: [{x0}, {y0}, 0.0]\n"
                    f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
        self.saved_version = self.map.version
        self.last_save = time.time()
        n = len(self.map.mines(True))
        self.get_logger().info(f"{n} mines confirmed, {len(self.map.tracks) - n} tentative, "
                               f"{self.map.coverage_area():.1f} m2 seen | frames {self.stats} | "
                               f"ignored on other drones {self.ignored}", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = MineMapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):     # Ctrl-C / SIGTERM
        pass
    except Exception:
        if rclpy.ok():
            raise                   # a real error, not the executor racing the shutdown
    finally:
        node.save(node.map.render(drones=node.positions, truth=node.truth))    # final save
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
