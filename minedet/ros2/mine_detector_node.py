#!/usr/bin/env python3
"""
ROS 2 node: mine detection and geo-location for one drone's downward camera.

Every processed frame is paired with the drone's pose at exposure time (interpolated
from the pose stream; attitude from the faster IMU stream if available), the
detections are projected onto the ground in the arena frame (tilt compensated,
lens distortion removed) and published as JSON for mine_map_node.py.

In the GPS-denied swarm the pose is the daughter's EKF output
(/NAME/mavros/local_position/pose), which fuses the mother's ArUco fixes and the
daughter's optical flow, so the mines land in the same arena frame the mother
estimates. No estimation happens on the daughter beyond her own autopilot.

Subscribes  image_topic, camera_info_topic (unless calib_file), pose_topic, attitude_topic,
            /mines/map (confirmed mines, drawn into the debug image)
Publishes   /NAME/mines/detections  std_msgs/String (JSON, one message per processed frame)
            /NAME/mines/debug_image sensor_msgs/Image

    python3 mine_detector_node.py --ros-args -p drone:=DeeDee
"""
import bisect
import collections
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import String

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from camera_model import CameraModel                                              # noqa: E402
from geo_projection import GroundProjector, camera_mount, quat_to_R               # noqa: E402
from mine_locator import MineLocator, draw_located                                # noqa: E402

try:
    from mavros_msgs.srv import MessageInterval
except ImportError:
    MessageInterval = None

GZ_CAMERA = "/world/iris_runway/model/{drone}/model/gimbal/link/pitch_link/sensor/camera"


def slerp(q0, q1, a):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


class Buffer:
    """Time-stamped samples; value at any time by linear interpolation (slerp for quaternions)."""

    def __init__(self, seconds=3.0, quaternion=False):
        self.t, self.v = collections.deque(), collections.deque()
        self.seconds, self.quat = seconds, quaternion
        self.lock = threading.Lock()

    def add(self, t, v):
        with self.lock:
            self.t.append(t)
            self.v.append(np.asarray(v, float))
            while self.t and self.t[0] < t - self.seconds:
                self.t.popleft()
                self.v.popleft()

    def newest(self):
        with self.lock:
            return self.t[-1] if self.t else None

    def at(self, t, max_gap=0.5, max_extrapolate=0.15):
        with self.lock:
            ts, vs = list(self.t), list(self.v)
        if not ts:
            return None
        i = bisect.bisect_left(ts, t)
        if i == 0:
            return vs[0] if ts[0] - t <= max_extrapolate else None
        if i == len(ts):
            return vs[-1] if t - ts[-1] <= max_extrapolate else None
        t0, t1 = ts[i - 1], ts[i]
        if t1 - t0 > max_gap:
            return None
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return slerp(vs[i - 1], vs[i], a) if self.quat else vs[i - 1] + a * (vs[i] - vs[i - 1])

    def rate(self, window=2.0):
        with self.lock:
            if len(self.t) < 2:
                return 0.0
            recent = [x for x in self.t if x > self.t[-1] - window]
        return (len(recent) - 1) / window if len(recent) > 1 else 0.0


def image_to_bgr(msg):
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(msg.encoding)
    if ch is None:
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch)
    if msg.encoding == "rgb8":
        return a[..., ::-1].copy()
    if msg.encoding == "rgba8":
        return a[..., 2::-1].copy()
    if msg.encoding == "bgra8":
        return a[..., :3].copy()
    if msg.encoding == "mono8":
        return np.repeat(a, 3, axis=2)
    return a.copy()


def bgr_to_image(img, header):
    m = Image()
    m.header = header
    m.height, m.width = img.shape[:2]
    m.encoding, m.step = "bgr8", img.shape[1] * 3
    m.data = np.ascontiguousarray(img).tobytes()
    return m


class MineDetectorNode(Node):
    def __init__(self):
        super().__init__("mine_detector")
        p = self.declare_parameter
        self.drone = p("drone", "DeeDee").value
        fmt = lambda s: s.format(drone=self.drone)                                        # noqa: E731
        self.image_topic = fmt(p("image_topic", GZ_CAMERA + "/image").value)
        self.info_topic = fmt(p("camera_info_topic", GZ_CAMERA + "/camera_info").value)
        self.calib_file = p("calib_file", "").value
        self.pose_topic = fmt(p("pose_topic", "/{drone}/mavros/local_position/pose").value)
        self.att_topic = fmt(p("attitude_topic", "/{drone}/mavros/imu/data").value)
        self.out_topic = fmt(p("detections_topic", "/{drone}/mines/detections").value)
        self.debug_topic = fmt(p("debug_topic", "/{drone}/mines/debug_image").value)
        self.map_topic = p("map_topic", "/mines/map").value
        detector = p("detector", "yolo").value                 # yolo | color | hybrid
        model = p("model", "640x480").value
        threads = p("threads", 2).value
        conf = p("conf", 0.3).value
        # ground scale the model was trained at (pixels per metre at its input; 184 = a
        # 640x480 rpi_cam_v2 from 3 m). Frames far off it are rescaled: shrunk for the 1 m
        # landing check, tiled if mines would be tiny. 0 feeds frames unchanged.
        ref_px_per_m = p("model_px_per_m", 184.0).value
        image_top = p("image_top", "forward").value
        offset = list(p("camera_offset", [0.0, -0.01, -0.1249]).value)
        stabilized = p("stabilized", False).value
        ground_z = p("ground_z", 0.0).value
        self.min_height = p("min_height", 0.7).value           # camera above ground (m): skip take-off / landed
        self.max_tilt = math.radians(p("max_tilt_deg", 30.0).value)
        self.max_rate = p("max_rate", 5.0).value               # processed frames per second
        self.image_delay = p("image_delay", 0.0).value         # s: image exposure precedes its arrival by this
        self.time_source = p("time_source", "receipt").value   # receipt | header (same clock as the poses)
        self.publish_debug = p("publish_debug", True).value
        self.request_rates = p("request_rates", True).value

        # camera model: calibration file now, or the first camera_info
        self.camera = CameraModel.from_file(self.calib_file) if self.calib_file else None
        self.projector_args = dict(R_body_cam=camera_mount(image_top), t_body_cam=offset, ground_z=ground_z,
                                   stabilized=stabilized)
        self.locator = None
        self.locator_args = dict(detector=detector, model=model, threads=threads, conf=conf,
                                 ref_px_per_m=ref_px_per_m)
        if self.camera is not None:
            self._make_locator()

        self.poses = Buffer(3.0)
        self.quats = Buffer(3.0, quaternion=True)
        self.att = Buffer(3.0, quaternion=True)
        self.map_mines = []
        self.pending = None
        self.cv = threading.Condition()
        self.last_accept = 0.0
        self.stats = collections.Counter()
        self.lat = collections.deque(maxlen=50)

        self.pub = self.create_publisher(String, self.out_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 2)
        if not self.calib_file:
            self.create_subscription(CameraInfo, self.info_topic, self.on_info, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.pose_topic, self.on_pose, qos_profile_sensor_data)
        if self.att_topic:
            self.create_subscription(Imu, self.att_topic, self.on_att, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, self.map_topic, self.on_map, latched)
        self.create_subscription(Image, self.image_topic, self.on_image, qos_profile_sensor_data)
        if self.request_rates and MessageInterval is not None:
            self.rate_cli = self.create_client(MessageInterval, f"/{self.drone}/mavros/set_message_interval")
            self.create_timer(2.0, self.keep_rates)
        self.create_timer(10.0, self.report)
        threading.Thread(target=self.worker, daemon=True).start()
        self.get_logger().info(f"{self.drone}: image {self.image_topic}, pose {self.pose_topic}, "
                               f"attitude {self.att_topic or 'from pose'} -> {self.out_topic}")

    def _make_locator(self):
        proj = GroundProjector(self.camera, **self.projector_args)
        try:
            self.locator = MineLocator(proj, **self.locator_args)
        except ImportError as e:
            self.get_logger().warn(f"{e}\n -> falling back to the colour/shape detector")
            self.locator = MineLocator(proj, **dict(self.locator_args, detector="color"))
        self.get_logger().info(f"detector {self.locator.describe()} | camera {self.camera.describe()}")

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def now():
        return time.monotonic()

    def stamp_time(self, header):
        """Our clock for a message: receipt time, or its header stamp (same clock as the poses)."""
        if self.time_source == "header":
            return header.stamp.sec + header.stamp.nanosec * 1e-9
        return self.now()

    def on_info(self, msg):
        if self.camera is None and msg.width > 0 and msg.k[0] > 0:
            self.camera = CameraModel.from_camera_info(msg)
            self._make_locator()

    def on_pose(self, msg):
        t = self.stamp_time(msg.header)
        p, q = msg.pose.position, msg.pose.orientation
        self.poses.add(t, (p.x, p.y, p.z))
        self.quats.add(t, (q.x, q.y, q.z, q.w))

    def on_att(self, msg):
        q = msg.orientation
        if q.w == 0 and q.x == 0 and q.y == 0 and q.z == 0:
            return
        self.att.add(self.stamp_time(msg.header), (q.x, q.y, q.z, q.w))

    def on_map(self, msg):
        try:
            self.map_mines = json.loads(msg.data).get("mines", [])
        except ValueError:
            pass

    def on_image(self, msg):
        t = self.stamp_time(msg.header) - self.image_delay
        self.stats["frames"] += 1
        if self.locator is None:
            return
        if self.now() - self.last_accept < 1.0 / max(self.max_rate, 1e-3):
            return
        self.last_accept = self.now()
        with self.cv:
            if self.pending is not None:
                self.stats["dropped_busy"] += 1
            self.pending = (msg, t)
            self.cv.notify()

    def keep_rates(self):
        """Same rates the swarm mission asks for (no tug of war): LOCAL_POSITION_NED 10 Hz,
        ATTITUDE_QUATERNION 20 Hz; only re-requested when a stream is slow."""
        if not self.rate_cli.service_is_ready():
            return
        for msg_id, hz, buf in ((32, 10.0, self.poses), (31, 20.0, self.att)):
            if buf.rate() < hz * 0.6:
                self.rate_cli.call_async(MessageInterval.Request(message_id=msg_id, message_rate=hz))

    # ---------------------------------------------------------------- processing
    def pose_at(self, t):
        p = self.poses.at(t)
        q = self.att.at(t) if self.att_topic else None
        if q is None:
            q = self.quats.at(t)
        if p is None or q is None:
            return None
        return p, quat_to_R(*q)

    def worker(self):
        while rclpy.ok():
            with self.cv:
                while self.pending is None:
                    self.cv.wait(0.5)
                    if not rclpy.ok():
                        return
                msg, t = self.pending
                self.pending = None
            try:
                img = image_to_bgr(msg)
            except ValueError as e:
                self.get_logger().error(str(e), throttle_duration_sec=10.0)
                continue
            # the pose stream lags a little: give it up to 0.2 s to pass the exposure time
            deadline = self.now() + 0.2
            while (self.poses.newest() or 0) < t and self.now() < deadline:
                time.sleep(0.01)
            self.process(img, t, msg.header)

    def process(self, img, t, header):
        pose = self.pose_at(t)
        if pose is None:
            self.stats["no_pose"] += 1
            return
        p, R = pose
        cam_z = self.locator.projector.camera_centre(p, R)[2] - self.locator.projector.ground_z
        tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
        if cam_z < self.min_height:
            self.stats["low"] += 1
            return
        if tilt > self.max_tilt:
            self.stats["tilted"] += 1
            return
        dets, footprint, ms = self.locator.locate(img, p, R)
        self.lat.append(ms)
        self.stats["processed"] += 1
        good = [d for d in dets if d["valid"]]
        self.stats["detections"] += len(good)
        q = self.quats.at(t)
        out = {
            "stamp": t, "drone": self.drone, "frame": "arena",
            # the camera's own time (sim time in Gazebo): lets an evaluator find the true pose
            "image_stamp": header.stamp.sec + header.stamp.nanosec * 1e-9,
            "pose": [round(float(a), 4) for a in list(p) + list(q if q is not None else (0, 0, 0, 1))],
            "height": round(float(cam_z), 3), "tilt_deg": round(math.degrees(tilt), 2),
            "footprint": [[round(float(x), 3), round(float(y), 3)] for x, y in footprint],
            "latency_ms": round(ms, 1), "rejected": len(dets) - len(good),
            "detections": [{
                "x": round(d["x"], 4), "y": round(d["y"], 4), "sigma": round(d["sigma"], 4),
                "class_name": d["class_name"], "confidence": round(float(d["confidence"]), 3),
                "size": round(d["size"], 3) if np.isfinite(d["size"]) else None,
                "u": round(d["u"], 1), "v": round(d["v"], 1), "box": [int(a) for a in d["box"]],
                "truncated": d["truncated"], "source": d.get("source", "")} for d in good],
        }
        self.pub.publish(String(data=json.dumps(out)))
        if self.publish_debug and self.debug_pub.get_subscription_count() > 0:
            overlay = []
            if self.map_mines:
                pts = np.array([[m["x"], m["y"], self.locator.projector.ground_z] for m in self.map_mines])
                uv = self.locator.projector.arena_to_pixels(pts, p, R)
                overlay = [(u, v, f"#{m['id']}") for (u, v), m in zip(uv, self.map_mines)]
            hud = [f"{self.drone} ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f}) tilt {math.degrees(tilt):.1f} deg | "
                   f"{len(good)} mines in view | {self.locator.describe()} {ms:.0f} ms"]
            self.debug_pub.publish(bgr_to_image(draw_located(img, dets, overlay, hud), header))

    def report(self):
        s = self.stats
        lat = f"{np.mean(self.lat):.0f} ms" if self.lat else "-"
        self.get_logger().info(
            f"{self.drone}: frames {s['frames']}, processed {s['processed']} ({lat}), detections {s['detections']}, "
            f"skipped: no pose {s['no_pose']}, low {s['low']}, tilted {s['tilted']}, busy {s['dropped_busy']} | "
            f"pose {self.poses.rate():.0f} Hz, attitude {self.att.rate():.0f} Hz")


def main(args=None):
    rclpy.init(args=args)
    node = MineDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):     # Ctrl-C / SIGTERM
        pass
    except Exception:
        if rclpy.ok():
            raise                   # a real error, not the executor racing the shutdown
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
