"""
Image -> ground projection: where on the arena floor is the thing at pixel (u, v)?

Frames (REP-103, the same as MAVROS and the ArUco localizer)
  arena   x, y horizontal, z up. In the GPS-denied swarm this is the ArUco arena
          frame: origin on the ground under the reference drone's marker.
  body    x forward, y left, z up (FLU). A MAVROS pose orientation is body -> arena.
  optical x = image right, y = image down, z = out of the lens (OpenCV).

For a pixel (u, v) (manual section C, "3D ray-plane ground projection with drone
tilt compensation", written in ENU/FLU):
  r_cam   = undistorted ray [x, y, 1]                  (CameraModel.pixels_to_rays)
  r_arena = R_arena_body @ R_body_cam @ r_cam          (full roll / pitch / yaw)
  C       = p_body + R_arena_body @ t_body_cam         (lens position)
  s       = (ground_z - C_z) / r_arena_z               (ray meets the ground plane)
  P       = C + s * r_arena
The nadir-only formula this replaces (X = -(dv/fy) h, Y = +(du/fx) h, then yaw) is
the special case roll = pitch = 0 with the lens at the body origin.
"""
import math
from collections import namedtuple

import numpy as np

# rpi_cam_v2 gimbal at rest (sim and the real mount): optical axis straight down,
# image right = drone right, image top = drone nose. Same as aruco_localizer.R_BODY_CAM.
R_BODY_CAM_DOWN = np.array([[0.0, -1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0]])
# lens position in the body frame for that mount (m)
T_BODY_CAM_RPI_GIMBAL = (0.0, -0.01, -0.1249)

GroundHits = namedtuple("GroundHits", "points valid range sigma off_nadir")


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_R(x, y, z, w):
    """Quaternion (x, y, z, w) -> rotation matrix."""
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy_to_R(roll, pitch, yaw):
    """REP-103 body (FLU) -> arena (ENU): R = Rz(yaw) Ry(pitch) Rx(roll).
    Positive pitch is nose DOWN here (MAVROS: pitch_enu = -pitch_ned)."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return rot_z(yaw) @ Ry @ Rx


def yaw_of_R(R):
    return math.atan2(R[1, 0], R[0, 0])


def camera_mount(image_top="forward"):
    """Downward camera turned about its optical axis: image top towards the drone's
    'forward' (default), 'left', 'backward' or 'right', or an angle in degrees (CCW)."""
    angles = {"forward": 0.0, "left": 90.0, "backward": 180.0, "right": -90.0}
    a = angles[image_top] if image_top in angles else float(image_top)
    return rot_z(math.radians(a)) @ R_BODY_CAM_DOWN


class GroundProjector:
    """Pixels -> points on the ground plane z = ground_z (arena frame).

    camera      CameraModel (intrinsics + distortion)
    R_body_cam  optical -> body rotation (default: downward rpi_cam_v2 mount)
    t_body_cam  lens position in the body frame (m)
    stabilized  True for a roll/pitch-stabilised gimbal (only yaw reaches the camera)
    sigma_*     1-sigma pixel, attitude and position errors used for each point's
                horizontal uncertainty (the map weights observations with it)
    """

    def __init__(self, camera, R_body_cam=None, t_body_cam=T_BODY_CAM_RPI_GIMBAL, ground_z=0.0,
                 stabilized=False, max_off_nadir_deg=80.0, max_ground_range=30.0,
                 sigma_px=2.0, sigma_att_deg=1.0, sigma_pos=0.03):
        self.camera = camera
        self.R_body_cam = R_BODY_CAM_DOWN if R_body_cam is None else np.asarray(R_body_cam, float).reshape(3, 3)
        self.t_body_cam = np.asarray(t_body_cam, dtype=float).ravel()
        self.ground_z = float(ground_z)
        self.stabilized = stabilized
        self.cos_max = math.cos(math.radians(max_off_nadir_deg))
        self.max_ground_range = max_ground_range
        self.sigma_px, self.sigma_att, self.sigma_pos = sigma_px, math.radians(sigma_att_deg), sigma_pos

    # ------------------------------------------------------------------ geometry
    def _attitude(self, R_arena_body):
        R = np.asarray(R_arena_body, dtype=float)
        return rot_z(yaw_of_R(R)) if self.stabilized else R

    def camera_centre(self, position, R_arena_body):
        return np.asarray(position, dtype=float) + np.asarray(R_arena_body, float) @ self.t_body_cam

    def R_arena_cam(self, R_arena_body):
        return self._attitude(R_arena_body) @ self.R_body_cam

    def project(self, uv, position, R_arena_body, ground_z=None):
        """(N, 2) pixels -> GroundHits: points (N, 3) on the plane, valid (N,) (ray hits the
        ground in front of the lens, not too close to the horizon), range (N,) lens -> point (m),
        sigma (N,) horizontal 1-sigma (m), off_nadir (N,) ray angle from straight down (rad)."""
        gz = self.ground_z if ground_z is None else ground_z
        rays = self.camera.pixels_to_rays(uv) @ self.R_arena_cam(R_arena_body).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        C = self.camera_centre(position, R_arena_body)
        down = -rays[:, 2]
        valid = down > max(self.cos_max, 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(valid, (C[2] - gz) / np.where(valid, down, 1.0), np.nan)
        valid &= s > 0
        pts = C + s[:, None] * rays
        pts[~valid] = np.nan
        f = 0.5 * (self.camera.fx + self.camera.fy)
        cos_t = np.clip(down, 1e-3, 1.0)
        ang = np.hypot(self.sigma_px / f, self.sigma_att)
        sigma = np.hypot(self.sigma_pos, s / cos_t * ang)
        return GroundHits(pts, valid, s, sigma, np.arccos(np.clip(down, -1.0, 1.0)))

    def project_boxes(self, boxes, position, R_arena_body, ground_z=None):
        """Detection boxes [x1, y1, x2, y2] -> (GroundHits of the box centres,
        (N, 2) ground size [across u, across v] in metres, NaN where unknown)."""
        b = np.asarray(boxes, dtype=float).reshape(-1, 4)
        cu, cv = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        pts = np.concatenate([np.stack([cu, cv], 1),
                              np.stack([b[:, 0], cv], 1), np.stack([b[:, 2], cv], 1),
                              np.stack([cu, b[:, 1]], 1), np.stack([cu, b[:, 3]], 1)])
        hits = self.project(pts, position, R_arena_body, ground_z)
        n = len(b)
        P = hits.points.reshape(5, n, 3)
        size = np.stack([np.linalg.norm(P[2] - P[1], axis=1), np.linalg.norm(P[4] - P[3], axis=1)], 1)
        centre = GroundHits(*(a[:n] for a in hits))
        return centre, size

    def footprint(self, position, R_arena_body, per_edge=6, ground_z=None):
        """Ground outline of the image, (M, 2) arena x, y (clockwise from image top-left).
        Border rays that miss the ground are cut at max_ground_range from the lens."""
        gz = self.ground_z if ground_z is None else ground_z
        uv = self.camera.border_pixels(per_edge)
        hits = self.project(uv, position, R_arena_body, gz)
        C = self.camera_centre(position, R_arena_body)
        out = []
        rays = self.camera.pixels_to_rays(uv) @ self.R_arena_cam(R_arena_body).T
        for p, ok, r in zip(hits.points, hits.valid, rays):
            if ok and math.hypot(p[0] - C[0], p[1] - C[1]) <= self.max_ground_range:
                out.append(p[:2])
            else:
                h = math.hypot(r[0], r[1])
                if h > 1e-9:
                    out.append(C[:2] + r[:2] / h * self.max_ground_range)
        return np.array(out).reshape(-1, 2)

    def arena_to_pixels(self, pts, position, R_arena_body):
        """(N, 3) arena points -> (N, 2) pixels (NaN behind the lens). For overlays."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        R = self.R_arena_cam(R_arena_body)
        pc = (pts - self.camera_centre(position, R_arena_body)) @ R
        uv = np.full((len(pts), 2), np.nan)
        front = pc[:, 2] > 1e-6
        if np.any(front):
            uv[front] = self.camera.rays_to_pixels(pc[front])
        return uv


def point_in_polygon(pts, poly):
    """(N, 2) points, (M, 2) polygon -> (N,) bool (even-odd rule)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    inside = np.zeros(len(pts), dtype=bool)
    if len(poly) < 3:
        return inside
    x, y = pts[:, 0], pts[:, 1]
    xj, yj = poly[-1]
    for xi, yi in poly:
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < xc)
        xj, yj = xi, yi
    return inside


def shrink_polygon(poly, margin):
    """Pull every vertex of a (convex-ish) polygon towards its centroid by `margin` metres."""
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(poly) == 0 or margin <= 0:
        return poly
    c = poly.mean(axis=0)
    d = poly - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return c + d * np.clip((n - margin) / np.maximum(n, 1e-9), 0.0, 1.0)
