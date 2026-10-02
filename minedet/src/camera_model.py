"""
Camera intrinsics and lens distortion: pixel <-> ray in the camera optical frame.

Optical frame (OpenCV / ROS): x = image right, y = image down, z = out of the lens.
Rays are unit vectors, so fisheye rays beyond 90 deg off-axis are fine too.

Supported lens models
  pinhole  - OpenCV plumb_bob / rational_polynomial (k1 k2 p1 p2 [k3 [k4 k5 k6]])
  fisheye  - OpenCV fisheye = ROS "equidistant" (k1 k2 k3 k4); use it for the
             160 deg lens: the pinhole model cannot fit that much distortion

Sources
  CameraModel.from_camera_info(msg)   sensor_msgs/CameraInfo (sim or camera driver)
  CameraModel.from_file(path)         ROS camera_calibration YAML, OpenCV YAML (both with
                                      image_width / image_height) or JSON
  CameraModel.from_hfov(w, h, hfov)   ideal pinhole (no calibration yet)
"""
import json
import math

import cv2
import numpy as np

FISHEYE_NAMES = ("fisheye", "equidistant", "kannala_brandt")


class CameraModel:
    def __init__(self, width, height, K, D=None, model="pinhole"):
        self.width = int(width)
        self.height = int(height)
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.model = "fisheye" if str(model).lower() in FISHEYE_NAMES else "pinhole"
        D = np.zeros(4 if self.model == "fisheye" else 5) if D is None or len(D) == 0 else D
        self.D = np.asarray(D, dtype=np.float64).ravel()
        if self.model == "fisheye":
            self.D = np.pad(self.D, (0, max(0, 4 - len(self.D))))[:4]

    # ----------------------------------------------------------------- constructors
    @classmethod
    def from_hfov(cls, width, height, hfov_rad):
        f = (width / 2.0) / math.tan(hfov_rad / 2.0)
        K = [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]]
        return cls(width, height, K)

    @classmethod
    def from_camera_info(cls, msg):
        """sensor_msgs/CameraInfo (ROS 2 field names k, d; ROS 1 names K, D also work)."""
        K = getattr(msg, "k", None)
        K = getattr(msg, "K", None) if K is None else K
        D = getattr(msg, "d", None)
        D = getattr(msg, "D", None) if D is None else D
        return cls(msg.width, msg.height, list(K), list(D or []), msg.distortion_model or "pinhole")

    @classmethod
    def from_file(cls, path):
        """ROS camera_calibration YAML (camera_matrix / distortion_coefficients / distortion_model),
        OpenCV FileStorage YAML (camera_matrix / distortion_coefficients as !!opencv-matrix),
        or JSON {width, height, K, D, model}."""
        if path.lower().endswith(".json"):
            d = json.load(open(path))
            return cls(d["width"], d["height"], d["K"], d.get("D"), d.get("model", "pinhole"))
        try:                                   # ROS camera_calibration layout (plain YAML)
            import yaml
            d = yaml.safe_load(open(path))
            if isinstance(d, dict) and isinstance(d.get("camera_matrix"), dict):
                return cls(d["image_width"], d["image_height"], d["camera_matrix"]["data"],
                           (d.get("distortion_coefficients") or {}).get("data"),
                           d.get("distortion_model", "pinhole"))
        except ImportError:
            pass
        except Exception:                      # OpenCV tags (!!opencv-matrix) are not plain YAML
            pass
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        try:
            node = fs.getNode("camera_matrix")
            if not fs.isOpened() or node.empty():
                raise ValueError(f"{path}: no camera_matrix")
            K, D = node.mat(), fs.getNode("distortion_coefficients").mat()
            model = fs.getNode("distortion_model").string() or fs.getNode("model").string() or "pinhole"
            w, h = int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real())
        finally:
            fs.release()
        return cls(w, h, K, D, model)

    # --------------------------------------------------------------------- helpers
    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    def scaled_to(self, width, height):
        """Same lens at another resolution (calibrated at one size, streaming at another)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return CameraModel(width, height, K, self.D, self.model)

    # ------------------------------------------------------------- lens models
    # Plain numpy (same results with OpenCV 4.x on the Pi and 5.x): OpenCV's
    # undistortPoints / fisheye.undistortPoints lose accuracy near the edge of wide lenses.
    def _distort_pinhole(self, x, y):
        k = np.pad(self.D, (0, max(0, 8 - len(self.D))))
        k1, k2, p1, p2, k3, k4, k5, k6 = k[:8]
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd

    def _theta_d(self, th):
        k1, k2, k3, k4 = self.D[:4]
        t2 = th * th
        return th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _normalised(self, uv):
        """pixels -> distorted normalised coordinates (skew handled)."""
        K = self.K
        yd = (uv[:, 1] - K[1, 2]) / K[1, 1]
        xd = (uv[:, 0] - K[0, 2] - K[0, 1] * yd) / K[0, 0]
        return xd, yd

    def pixels_to_rays(self, uv):
        """(N, 2) pixels -> (N, 3) unit rays in the optical frame (distortion removed).
        Fisheye rays may point more than 90 deg off the optical axis (z <= 0)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if len(uv) == 0:
            return np.zeros((0, 3))
        xd, yd = self._normalised(uv)
        if self.model == "fisheye":
            rd = np.hypot(xd, yd)
            th = rd.copy()
            for _ in range(20):                     # Newton: theta_d(th) = rd
                t2 = th * th
                k1, k2, k3, k4 = self.D[:4]
                f = self._theta_d(th) - rd
                df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
                th = th - f / np.where(np.abs(df) > 1e-9, df, 1e-9)
            phi = np.arctan2(yd, xd)
            rays = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], 1)
        else:
            x, y = xd.copy(), yd.copy()
            if np.any(self.D):
                for _ in range(8):                  # fixed point to get close ...
                    px, py = self._distort_pinhole(x, y)
                    x, y = x + (xd - px), y + (yd - py)
                e = 1e-6
                for _ in range(6):                  # ... Gauss-Newton to finish
                    px, py = self._distort_pinhole(x, y)
                    ax, ay = self._distort_pinhole(x + e, y)
                    bx, by = self._distort_pinhole(x, y + e)
                    j11, j21, j12, j22 = (ax - px) / e, (ay - py) / e, (bx - px) / e, (by - py) / e
                    det = j11 * j22 - j12 * j21
                    det = np.where(np.abs(det) > 1e-12, det, 1e-12)
                    rx, ry = xd - px, yd - py
                    x, y = x + (j22 * rx - j12 * ry) / det, y + (j11 * ry - j21 * rx) / det
            rays = np.stack([x, y, np.ones_like(x)], 1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def rays_to_pixels(self, rays):
        """(N, 3) points / rays in the optical frame -> (N, 2) pixels (NaN where the lens
        cannot see them: behind a pinhole camera)."""
        p = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0:
            return np.zeros((0, 2))
        X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
        if self.model == "fisheye":
            r = np.hypot(X, Y)
            th = np.arctan2(r, Z)
            scale = np.where(r > 1e-12, self._theta_d(th) / np.maximum(r, 1e-12), 1.0 / np.where(Z > 0, Z, 1.0))
            xd, yd = X * scale, Y * scale
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                x, y = np.where(Z > 1e-9, X / Z, np.nan), np.where(Z > 1e-9, Y / Z, np.nan)
            xd, yd = self._distort_pinhole(x, y)
        K = self.K
        return np.stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)

    def border_pixels(self, per_edge=6, inset=0.5):
        """Pixels around the image border, clockwise from the top-left corner."""
        w, h, n = self.width - inset, self.height - inset, per_edge
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        top = np.stack([inset + t * (w - inset), np.full(n, inset)], 1)
        right = np.stack([np.full(n, w), inset + t * (h - inset)], 1)
        bottom = np.stack([w - t * (w - inset), np.full(n, h)], 1)
        left = np.stack([np.full(n, inset), h - t * (h - inset)], 1)
        return np.vstack([top, right, bottom, left])

    def describe(self):
        return (f"{self.model} {self.width}x{self.height} fx {self.fx:.1f} fy {self.fy:.1f} "
                f"cx {self.K[0, 2]:.1f} cy {self.K[1, 2]:.1f} D {np.round(self.D, 4).tolist()}")
