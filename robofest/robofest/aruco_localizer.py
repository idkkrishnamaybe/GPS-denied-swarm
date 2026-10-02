#!/usr/bin/env python3
"""ArUco "GPS": localize every drone from the mother drone's downward camera.

Each drone carries an ArUco marker on top. Landed drones are the fixed
references ("landmarks"). The mother hovers high, sees a landmark and the other
drones' markers, and computes everyone's pose in the *arena* frame, defined by
the first reference (`reference_id`):

    origin  on the ground directly below the first reference marker
    +x      its nose direction,  +y its left,  +z up

Leapfrogging: the mission publishes the landed drones on /aruco/landmarks
(std_msgs/String, e.g. "12,11", in order of preference). A newly listed drone is
frozen at its measured arena pose (averaged over the last seconds, while an older
landmark was still the anchor), so the arena frame carries over from landmark to
landmark as the swarm moves across the field. Each frame is anchored on the first
visible landmark; landmarks get their frozen pose published (their autopilots
need it too).

Outputs, per drone NAME in `markers`:
    /NAME/aruco/pose                 geometry_msgs/PoseStamped (frame "arena"), body pose
    /NAME/mavros/vision_pose/pose    same pose, only if `publish_mavros` is true (GPS replacement)
    /aruco/debug_image               annotated, downscaled camera image (if `debug_image`)

Geometry: roll/pitch of the mother's camera come from her autopilot IMU
(`attitude_topic`, MAVROS imu/data); the markers give bearing + range and
headings. A small marker's own tilt is only known to a few degrees, which at
7 m would mean tens of cm of error, so it is not used while the IMU is fresh.
Without IMU data the node falls back to marker-only PnP (much less accurate).

Measured in Gazebo (mother 6-8 m, scanner 3-4 m, 18 cm markers, 1640x1232):
~1-2 cm horizontal / ~3 cm vertical error, <0.2 deg heading.
"""
import signal
import time

import numpy as np
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from collections import deque

from sensor_msgs.msg import Image, CameraInfo, Imu
from std_msgs.msg import Float64, String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
try:
    from mavros_msgs.srv import MessageInterval
except ImportError:          # MAVROS not installed: attitude-rate request disabled
    MessageInterval = None

# Camera optical frame (x right, y down, z forward) expressed in the mother's
# body frame (x forward, y left, z up) for the rpi_cam_v2 gimbal at rest:
# image right = drone right, image down = drone tail, optical axis = straight down.
R_BODY_CAM = np.array([[0.0, -1.0, 0.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0]])

# Marker frame (x = marker right, y = marker top, z = out of the plate) in the
# carrying drone's body frame. Marker top points at the drone's nose.
R_BODY_MARKER = np.array([[0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def R_from_quat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_tf(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_tf(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_tf(R.T, -R.T @ t)


def quat_from_R(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return x, y, z, w


def aruco_dictionary(name):
    const = getattr(cv2.aruco, name)
    try:
        return cv2.aruco.getPredefinedDictionary(const)
    except AttributeError:
        return cv2.aruco.Dictionary_get(const)


def refine_corners_edges(gray, corners, samples=24, half_width=3.0):
    """Re-locate the 4 marker corners by fitting lines to sub-pixel edge points.

    For each side, sample points along it, find the peak of the intensity
    gradient across the side (parabolic sub-pixel fit), fit a line, and
    intersect neighbouring lines. Unlike threshold/contour corners, the gradient
    peak sits on the true edge for any symmetric blur, so there is no inward bias.
    Returns the input corners unchanged if a side cannot be fitted.
    """
    # Work in linear light: cameras and renderers blur edges in linear intensity and
    # then gamma-encode (sRGB), which shifts the visible edge into the dark side.
    v = gray.astype(np.float32) / 255.0
    img = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4) * 255.0
    h, w = img.shape
    c = corners.astype(np.float64)
    center = c.mean(0)
    lines = []
    offs = np.linspace(-half_width, half_width, int(half_width * 4) + 1)
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        if L < 8:
            return corners
        t = d / L
        n = np.array([-t[1], t[0]])
        if np.dot(n, center - a) < 0:        # normal points outwards (away from the marker)
            n = -n
        n = -n
        pts = []
        for s in np.linspace(0.15, 0.85, samples):
            p = a + s * d
            q = p[None, :] + offs[:, None] * n[None, :]
            if q.min() < 1 or q[:, 0].max() > w - 2 or q[:, 1].max() > h - 2:
                continue
            prof = cv2.remap(img, q[:, 0:1].astype(np.float32), q[:, 1:2].astype(np.float32),
                             cv2.INTER_LINEAR).ravel()
            g = np.abs(np.diff(prof))
            i = int(np.argmax(g))
            if i == 0 or i == len(g) - 1 or g[i] < 3:
                continue
            den = g[i - 1] - 2 * g[i] + g[i + 1]
            sub = 0.5 * (g[i - 1] - g[i + 1]) / den if den != 0 else 0.0
            o = (offs[i] + offs[i + 1]) / 2 + sub * (offs[1] - offs[0])
            pts.append(p + o * n)
        if len(pts) < samples // 3:
            return corners
        pts = np.array(pts)
        m = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - m)
        lines.append((m, vt[0]))
    out = []
    for k in range(4):
        (p1, d1), (p2, d2) = lines[(k - 1) % 4], lines[k]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return corners
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out)
    if np.max(np.linalg.norm(out - c, axis=1)) > 3.0:     # refinement went wrong; keep original
        return corners
    return out.astype(corners.dtype)


class MarkerDetector:
    """Same interface on OpenCV 4.6 (system python) and 4.7+/5.x (venv)."""

    def __init__(self, dict_name):
        self.dictionary = aruco_dictionary(dict_name)
        # OpenCV 4.6 has a DetectorParameters() constructor too, but it returns an
        # unusable object (writing a field segfaults); _create() is the real one there.
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.params = cv2.aruco.DetectorParameters_create()
        else:
            self.params = cv2.aruco.DetectorParameters()
        # Sub-pixel corners: the biggest single accuracy gain for small markers.
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize = 3
        self.detector = None
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, gray):
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        if ids is None:
            return {}
        return {int(i): refine_corners_edges(gray, c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)}


class ArucoLocalizer(Node):
    def __init__(self):
        super().__init__('aruco_localizer')
        p = self.declare_parameter
        self.image_topic = p('image_topic',
                             '/world/iris_runway/model/Joey/model/gimbal/link/pitch_link/sensor/camera/image').value
        self.info_topic = p('camera_info_topic',
                            '/world/iris_runway/model/Joey/model/gimbal/link/pitch_link/sensor/camera/camera_info').value
        dict_name = p('dictionary', 'DICT_5X5_100').value
        self.marker_size = p('marker_size', 0.18).value
        markers = p('markers', ['10:Joey', '11:DeeDee', '12:Marky']).value
        self.reference_id = p('reference_id', 12).value
        self.self_name = p('self_name', 'Joey').value
        # Height of the reference marker surface above the ground while its drone is landed
        # (sim iris: base_link rests at 0.195 m, marker surface 0.136 m above it).
        # Every z output is shifted by any error here: measure it on the real drone.
        self.reference_height = p('reference_marker_height', 0.331).value
        # Marker surface above each drone's body origin.
        self.marker_offset_z = p('marker_offset_z', 0.136).value
        # Camera optical centre in the mother's body frame (x fwd, y left, z up).
        self.camera_offset = np.array(p('camera_offset', [0.0, -0.01, -0.1249]).value, dtype=float)
        self.publish_mavros = p('publish_mavros', False).value
        self.vision_topic = p('vision_topic', '/{name}/mavros/vision_pose/pose').value
        self.pose_topic = p('pose_topic', '/{name}/aruco/pose').value
        self.debug_image = p('debug_image', True).value
        # Mother attitude (roll/pitch) from its autopilot. A distant 18 cm marker's tilt is
        # only known to a few degrees, which at 7 m means tens of cm of position error;
        # the IMU's roll/pitch is far better, so the marker only supplies the heading.
        # Empty string = vision only.
        self.attitude_topic = p('attitude_topic', '/{self}/mavros/imu/data').value
        self.attitude_timeout = p('attitude_timeout', 0.5).value
        # ArduPilot streams attitude at only ~2 Hz by default; ask for more over MAVROS.
        self.attitude_rate = p('request_attitude_hz', 50.0).value
        self.rate_service = p('message_interval_service', '/{self}/mavros/set_message_interval').value

        self.names = {}
        for m in markers:
            mid, name = m.split(':')
            self.names[int(mid)] = name
        if self.reference_id not in self.names:
            raise ValueError('reference_id must be listed in markers')

        s = self.marker_size / 2.0
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left.
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

        # Arena frame <- reference marker frame (reference drone body, lifted to marker height).
        self.T_arena_ref = make_tf(R_BODY_MARKER, [0.0, 0.0, self.reference_height])
        # Body <- marker for any carrying drone; camera <- mother body.
        self.T_marker_body = inv_tf(make_tf(R_BODY_MARKER, [0.0, 0.0, self.marker_offset_z]))
        self.T_cam_body = inv_tf(make_tf(R_BODY_CAM, self.camera_offset))
        # A landed drone's body in its own arena frame (on the ground below its marker).
        self.T_landmark_body = make_tf(np.eye(3), [0.0, 0.0, self.reference_height - self.marker_offset_z])

        # Landmarks: id -> T_arena_A, where A is that landmark's own frame (ground below its
        # marker, its body axes). The first reference defines the arena, so it is identity.
        self.landmarks = {self.reference_id: np.eye(4)}
        self.landmark_order = [self.reference_id]
        self.history = {}               # id -> recent (t, x, y, yaw) arena body poses
        self.anchor = None
        self.create_subscription(String, '/aruco/landmarks', self.on_landmarks, 10)
        # Test hook: /aruco/inject_dropout N publishes no fixes for N seconds (0 ends it early),
        # as if no marker were visible. Images keep being consumed, like a real dropout.
        self.blind_until = 0.0
        self.create_subscription(Float64, '/aruco/inject_dropout', self.on_inject_dropout, 10)
        # Landmarks actually in use (mission waits for this before releasing an old one).
        self.active_pub = self.create_publisher(String, '/aruco/landmarks_active', 10)

        self.detector = MarkerDetector(dict_name)
        self.bridge = CvBridge()
        self.K = None
        self.D = None

        self.pose_pubs = {}
        self.vision_pubs = {}
        for name in self.names.values():
            self.pose_pubs[name] = self.create_publisher(PoseStamped, self.pose_topic.format(name=name), 10)
            if self.publish_mavros:
                self.vision_pubs[name] = self.create_publisher(
                    PoseStamped, self.vision_topic.format(name=name), 10)
        self.debug_pub = self.create_publisher(Image, '/aruco/debug_image', 1) if self.debug_image else None

        self.R_level_body = None
        self.attitude_time = None
        if self.attitude_topic:
            self.create_subscription(Imu, self.attitude_topic.format(self=self.self_name),
                                     self.on_attitude, qos_profile_sensor_data)
        self.rate_client = None
        if self.attitude_topic and self.attitude_rate > 0 and MessageInterval is not None:
            self.rate_client = self.create_client(MessageInterval, self.rate_service.format(self=self.self_name))
            # Checked every second and re-sent whenever attitude goes stale: MAVProxy /
            # ground-station stream-rate requests can reset it at any time.
            self.rate_ok = False
            self.create_timer(1.0, self.request_attitude_rate)
        self.create_subscription(CameraInfo, self.info_topic, self.on_info, qos_profile_sensor_data)
        # Only the newest frame: after any stall (CPU, a slow frame) the queue would hand over
        # old images, which would be combined with the current attitude into wrong fixes.
        latest_only = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                 reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.image_topic, self.on_image, latest_only)
        self.frames = 0
        self.ref_seen = 0
        self.imu_used = 0
        self.proc_time = 0.0
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            f'markers {self.names}, reference {self.reference_id} ({self.names[self.reference_id]}), '
            f'self {self.self_name}, size {self.marker_size} m, mavros output {self.publish_mavros}')

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)
            self.get_logger().info(f'camera {msg.width}x{msg.height}, fx={self.K[0, 0]:.1f}')

    def request_attitude_rate(self):
        """Ask the autopilot for fast attitude messages at `request_attitude_hz`.

        ATTITUDE_QUATERNION (31) is not part of any stream group, so MAVProxy / GCS
        stream-rate requests do not reset it; ATTITUDE (30) is requested as a backup.
        MAVROS publishes imu/data orientation from either.
        """
        if not self.rate_client.service_is_ready():
            return
        if self.rate_ok and self.attitude_time is not None and \
                (self.get_clock().now() - self.attitude_time).nanoseconds < 0.2e9:
            return                                    # still streaming fast, nothing to do
        def done(fut):
            if fut.result() is not None and fut.result().success and not self.rate_ok:
                self.rate_ok = True
                self.get_logger().info(f'autopilot attitude stream set to {self.attitude_rate:g} Hz')
        for msg_id in (31, 30):
            req = MessageInterval.Request(message_id=msg_id, message_rate=float(self.attitude_rate))
            self.rate_client.call_async(req).add_done_callback(done)

    def on_attitude(self, msg):
        q = msg.orientation
        R = R_from_quat(q.x, q.y, q.z, q.w)          # world (ENU) <- body (FLU)
        self.R_level_body = rot_z(-np.arctan2(R[1, 0], R[0, 0])) @ R   # drop yaw: level <- body
        self.attitude_time = self.get_clock().now()

    def fresh_attitude(self):
        if self.R_level_body is None:
            return None
        age = (self.get_clock().now() - self.attitude_time).nanoseconds * 1e-9
        return self.R_level_body if age < self.attitude_timeout else None

    def on_landmarks(self, msg):
        ids = [int(v) for v in msg.data.replace(' ', '').split(',') if v]
        ids = [i for i in ids if i in self.names and self.names[i] != self.self_name]
        if not ids:
            self.get_logger().warning('ignoring empty landmark list', throttle_duration_sec=5)
            return
        for i in ids:
            if i not in self.landmarks:
                T = self.freeze(i)
                if T is not None:
                    self.landmarks[i] = T
        order = [i for i in ids if i in self.landmarks]
        if not order:
            return                          # new landmarks not ready yet; keep the old ones
        for i in list(self.landmarks):
            if i not in ids:
                del self.landmarks[i]       # it took off
                self.get_logger().info(f'landmark {self.names[i]} released')
        self.landmark_order = order

    def freeze(self, mid, window=3.0, min_samples=8, max_spread=0.05):
        """Arena frame of a just-landed drone, from its recent measured poses."""
        now = self.get_clock().now().nanoseconds * 1e-9
        h = [e for e in self.history.get(mid, ()) if now - e[0] < window]
        if len(h) < min_samples:
            self.get_logger().warning(f'not enough recent fixes to freeze {self.names[mid]} yet',
                                      throttle_duration_sec=2)
            return None
        a = np.array(h)
        if a[:, 1].std() > max_spread or a[:, 2].std() > max_spread:
            self.get_logger().warning(f'{self.names[mid]} is still moving; not frozen yet', throttle_duration_sec=2)
            return None
        x, y = a[:, 1].mean(), a[:, 2].mean()
        yaw = np.arctan2(np.sin(a[:, 3]).mean(), np.cos(a[:, 3]).mean())
        self.get_logger().info(f'landmark {self.names[mid]} frozen at ({x:+.3f}, {y:+.3f}) m, '
                               f'heading {np.degrees(yaw):+.1f} deg ({len(h)} fixes)')
        return make_tf(rot_z(yaw), [x, y, 0.0])

    def solve_marker(self, corners):
        """Marker pose in the camera frame (T_cam_marker).

        IPPE gives two candidate poses for a square; pick the one whose marker
        normal points back towards the camera most directly (markers lie roughly
        flat and the camera looks roughly straight down).
        """
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            self.obj_pts, corners.astype(np.float64), self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        best, best_score = None, None
        for rvec, tvec, err in zip(rvecs, tvecs, errs):
            R, _ = cv2.Rodrigues(rvec)
            facing = -R[2, 2]           # marker z vs. camera z; 1.0 = facing the camera head-on
            score = facing - 0.05 * float(np.ravel(err)[0])
            if best_score is None or score > best_score:
                best, best_score = make_tf(R, tvec), score
        return best

    def on_inject_dropout(self, msg):
        self.blind_until = time.monotonic() + max(0.0, msg.data)
        self.get_logger().warn(f'test: no ArUco fixes for {msg.data:.1f} s' if msg.data > 0 else
                               'test: ArUco fixes back')

    def on_image(self, msg):
        if self.K is None or time.monotonic() < self.blind_until:
            return
        t0 = time.monotonic()
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = {i: c for i, c in self.detector.detect(gray).items() if i in self.names}
        self.frames += 1

        poses = {}
        self.anchor = next((i for i in self.landmark_order if i in found), None)
        if self.anchor is not None:
            self.ref_seen += 1
            R_level_body = self.fresh_attitude()
            if R_level_body is not None:
                self.imu_used += 1
                local = self.poses_with_attitude(found, R_level_body, self.anchor)
            else:
                local = self.poses_vision_only(found, self.anchor)
            T_arena_A = self.landmarks[self.anchor]
            now = self.get_clock().now().nanoseconds * 1e-9
            ids = {v: k for k, v in self.names.items()}
            for name, T in local.items():
                poses[name] = T_arena_A @ T
                if name != self.self_name:
                    x, y = poses[name][0, 3], poses[name][1, 3]
                    yaw = np.arctan2(poses[name][1, 0], poses[name][0, 0])
                    self.history.setdefault(ids[name], deque(maxlen=100)).append((now, x, y, yaw))
            for mid, T_arena_L in self.landmarks.items():     # landed drones: frozen pose
                poses[self.names[mid]] = T_arena_L @ self.T_landmark_body

        for name, T in poses.items():
            out = PoseStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = 'arena'
            out.pose.position.x, out.pose.position.y, out.pose.position.z = (float(v) for v in T[:3, 3])
            q = quat_from_R(T[:3, :3])
            out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = q
            self.pose_pubs[name].publish(out)
            if self.publish_mavros:
                self.vision_pubs[name].publish(out)

        self.active_pub.publish(String(data=','.join(str(i) for i in self.landmark_order)))
        self.proc_time += time.monotonic() - t0
        if self.debug_pub is not None and self.debug_pub.get_subscription_count() > 0:
            self.publish_debug(img, found, poses, msg.header)

    def poses_vision_only(self, found, anchor):
        """Everything from marker PnP, including the camera tilt (fallback).
        Poses are in the anchor landmark's own frame."""
        T_cam_ref = self.solve_marker(found[anchor])
        T_arena_cam = self.T_arena_ref @ inv_tf(T_cam_ref)
        poses = {self.self_name: T_arena_cam @ self.T_cam_body}
        for mid, corners in found.items():
            name = self.names[mid]
            if mid not in self.landmarks and name != self.self_name:
                poses[name] = T_arena_cam @ self.solve_marker(corners) @ self.T_marker_body
        return poses

    def poses_with_attitude(self, found, R_level_body, anchor):
        """Camera tilt from the IMU; markers give positions (bearing + range) and headings.

        L is a gravity-aligned frame centred on the camera, x = mother's heading.
        Poses are in the anchor landmark's own frame.
        """
        R_lc = R_level_body @ R_BODY_CAM                  # L <- camera
        marker = {}
        for mid, corners in found.items():
            T = self.solve_marker(corners)
            nose = R_lc @ T[:3, 1]                        # marker top = carrying drone's nose
            marker[mid] = (R_lc @ T[:3, 3], np.arctan2(nose[1], nose[0]))
        p_ref, yaw_ref = marker[anchor]
        R_al = rot_z(-yaw_ref)                            # arena <- L
        up = np.array([0.0, 0.0, 1.0])

        def to_arena(p_l):
            return R_al @ (p_l - p_ref) + up * self.reference_height

        R_a_body = R_al @ R_level_body
        p_cam = to_arena(np.zeros(3))
        poses = {self.self_name: make_tf(R_a_body, p_cam - R_a_body @ self.camera_offset)}
        for mid, (p_l, yaw) in marker.items():
            name = self.names[mid]
            if mid not in self.landmarks and name != self.self_name:
                # other drones: position from the marker, heading from its top edge,
                # assumed near level (the 13.6 cm marker offset barely moves with tilt)
                poses[name] = make_tf(rot_z(yaw - yaw_ref), to_arena(p_l) - up * self.marker_offset_z)
        return poses

    def publish_debug(self, img, found, poses, header):
        for mid, c in found.items():
            color = (0, 200, 255) if mid in self.landmarks else (0, 255, 0)
            cv2.polylines(img, [c.astype(np.int32)], True, color, 3)
            label = self.names[mid] + (' (anchor)' if mid == self.anchor else ' (landed)' if mid in self.landmarks else '')
            if self.names[mid] in poses:
                x, y, z = poses[self.names[mid]][:3, 3]
                label += f' {x:+.2f},{y:+.2f},{z:.2f}'
            cv2.putText(img, label, tuple(int(v) for v in c[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if self.self_name in poses:
            x, y, z = poses[self.self_name][:3, 3]
            cv2.putText(img, f'{self.self_name} (self): {x:+.2f}, {y:+.2f}, {z:.2f} m', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)
        elif self.anchor is None:
            cv2.putText(img, 'NO LANDED REFERENCE VISIBLE', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        out = self.bridge.cv2_to_imgmsg(small, 'bgr8')
        out.header = header
        self.debug_pub.publish(out)

    def report(self):
        if self.frames:
            imu = f', IMU attitude used in {100 * self.imu_used // self.ref_seen}%' if self.ref_seen else ''
            lm = ','.join(self.names[i] for i in self.landmark_order)
            self.get_logger().info(
                f'{self.frames} frames ({1000 * self.proc_time / self.frames:.0f} ms each), '
                f'a landmark visible in {100 * self.ref_seen // self.frames}%{imu}; landmarks [{lm}]')
        self.frames = self.ref_seen = self.imu_used = 0
        self.proc_time = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocalizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # ros2 launch forwards Ctrl-C on top of the terminal's own; don't die mid-cleanup
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
