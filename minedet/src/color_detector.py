"""
Classical mine detector: colour / brightness anomalies against the local background,
filtered by shape and (when the altitude is known) by real size.

A fallback and a complement to the YOLO model: no training data needed, and it also
flags surface markers the model may miss (painted rings, disturbed soil, peg caps).
Same output format as YOLO26LiteRT.predict().

  disc / ellipse   compact, convex blob of mine size          -> mine_disc
  painted ring     blob with a hole                           -> surface_marker
  disturbed soil   irregular (non-convex) patch of mine size  -> surface_marker
  peg cap          small, very distinct dot                   -> surface_marker
"""
import math
import time

import cv2
import numpy as np


class ColorBlobDetector:
    def __init__(self, min_contrast=22.0, mine_size=(0.15, 0.45), peg_size=(0.03, 0.12),
                 px_size=(8, 160), downscale=4, conf_scale=1.0):
        self.min_contrast = min_contrast
        self.mine_size = mine_size          # accepted ground size of discs / rings / soil (m)
        self.peg_size = peg_size            # accepted ground size of a peg cap (m)
        self.px_size = px_size              # size limits in pixels when the scale is unknown
        self.downscale = downscale
        self.conf_scale = conf_scale
        self.class_names = {0: "mine_disc", 1: "surface_marker"}

    def _contrast(self, img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        h, w = lab.shape[:2]
        k = self.downscale
        small = cv2.resize(lab, (w // k, h // k), interpolation=cv2.INTER_AREA)
        ks = max(3, (min(small.shape[:2]) // 5) | 1)
        bg = cv2.resize(cv2.medianBlur(small, ks), (w, h), interpolation=cv2.INTER_LINEAR)
        d = lab.astype(np.float32) - bg.astype(np.float32)
        # brightness counts half: shadows and lighting gradients are not mines
        return np.sqrt(0.25 * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2) * 1.3

    def predict(self, img_bgr, px_per_m=None):
        """px_per_m: image scale on the ground (focal length / height), or None."""
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        diff = self._contrast(img_bgr)
        mask = (diff > self.min_contrast).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        if hier is None:
            return dets, (time.perf_counter() - t0) * 1000
        hier = hier[0]
        if px_per_m:
            lo, hi = self.mine_size[0] * px_per_m, self.mine_size[1] * px_per_m * 1.3
            plo, phi = self.peg_size[0] * px_per_m, self.peg_size[1] * px_per_m
        else:
            lo, hi = self.px_size
            plo, phi = 3, lo
        for i, c in enumerate(contours):
            if hier[i][3] != -1:                       # a hole, handled with its parent
                continue
            area = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            size = max(bw, bh)
            if area < 6 or size < plo or size > hi:
                continue
            hull = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull if hull > 0 else 0.0
            perim = cv2.arcLength(c, True)
            circ = 4 * math.pi * area / (perim * perim) if perim > 0 else 0.0
            holes = [j for j in range(len(contours)) if hier[j][3] == i]
            hole_area = sum(cv2.contourArea(contours[j]) for j in holes)
            m = np.zeros((bh, bw), np.uint8)
            cv2.drawContours(m, [c - [x, y]], -1, 1, -1)
            contrast = float(diff[y:y + bh, x:x + bw][m > 0].mean())
            elong = max(bw, bh) / max(1, min(bw, bh))
            cls, conf = None, 0.0
            if size <= phi and size < lo:                                  # peg cap / small dot
                if contrast > 2.2 * self.min_contrast and circ > 0.6:
                    cls, conf = 1, 0.45
            elif hole_area > 0.25 * hull:                                  # painted ring
                cls, conf = 1, 0.55 + 0.3 * min(1.0, hole_area / hull)
            elif solidity > 0.9 and circ > 0.72 and elong < 2.3:           # disc / ellipse
                cls, conf = 0, 0.5 + 0.35 * min(1.0, (circ - 0.72) / 0.2)
            elif solidity > 0.7 and elong < 2.0:                           # irregular patch
                cls, conf = 1, 0.35
            if cls is None:
                continue
            conf *= min(1.0, contrast / (2 * self.min_contrast)) * self.conf_scale
            dets.append({"class_id": cls, "class_name": self.class_names[cls], "confidence": round(conf, 3),
                         "box": [int(x), int(y), int(min(w, x + bw)), int(min(h, y + bh))], "source": "color"})
        return dets, (time.perf_counter() - t0) * 1000


def merge_detections(primary, secondary, iou_thresh=0.3):
    """Union of two detectors' boxes: a secondary box overlapping a primary one is dropped."""
    out = list(primary)
    for d in secondary:
        if all(box_overlap(d["box"], p["box"]) < iou_thresh for p in primary):
            out.append(d)
    return out


def box_overlap(a, b):
    """Intersection over the smaller box (1.0 when one box contains the other)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0
