"""
Detection + geo-location: one camera frame and the drone pose at exposure time ->
mine detections with arena coordinates, and the frame's ground footprint.

Used by the bench / CLI pipeline (test_pipeline.py) and the ROS 2 node
(ros2/mine_detector_node.py), so both give identical numbers.

detector: "yolo"   the trained YOLO26n LiteRT model (default)
          "color"  classical colour / shape detector (color_detector.py)
          "hybrid" YOLO plus the colour detector's extra finds (e.g. surface markers)
"""
import cv2
import numpy as np

from color_detector import ColorBlobDetector, merge_detections


def _contained(a, b):
    """Fraction of box a inside box b ([x1, y1, x2, y2])."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


class MineLocator:
    def __init__(self, projector, detector="yolo", model="640x480", threads=2, conf=0.3, iou=0.45,
                 min_size=0.06, max_size=0.8, truncated_sigma_scale=4.0, color_conf_scale=0.8,
                 ref_px_per_m=184.0, tile_overlap=0.15):
        self.projector = projector
        self.mode = detector
        self.yolo = None
        self.color = None
        if detector in ("yolo", "hybrid"):
            from detect import YOLO26LiteRT      # needs ai-edge-litert / tflite-runtime
            self.yolo = YOLO26LiteRT(model_path=model, threads=threads, conf_thresh=conf, iou_thresh=iou)
        if detector in ("color", "hybrid"):
            self.color = ColorBlobDetector(conf_scale=color_conf_scale if detector == "hybrid" else 1.0)
        if self.yolo is None and self.color is None:
            raise ValueError(f"unknown detector '{detector}' (yolo, color, hybrid)")
        self.conf = conf
        self.min_size, self.max_size = min_size, max_size
        self.truncated_sigma_scale = truncated_sigma_scale
        # ground scale the model knows (pixels per metre at its input; 184 = a daughter's
        # 640x480 camera from 3 m). Frames far off it are resized to it: shrunk and padded
        # (a landing check from 1 m) or, if mines would be tiny, tiled at full resolution.
        # 0 = feed every frame as it is.
        self.ref_px_per_m = ref_px_per_m
        self.tile_overlap = tile_overlap

    @property
    def camera(self):
        return self.projector.camera

    def describe(self):
        parts = []
        if self.yolo is not None:
            parts.append(f"YOLO {self.yolo.model_path.split('/')[-1]} ({self.yolo.in_w}x{self.yolo.in_h})")
        if self.color is not None:
            parts.append("colour/shape")
        return " + ".join(parts)

    def _yolo_rescaled(self, img_bgr, k):
        """YOLO on the frame resized by k: padded onto a neutral canvas if it became smaller
        than the model input, cut into overlapping input-sized tiles if larger. Boxes are
        mapped back to the full frame and duplicates from overlapping tiles merged."""
        h, w = img_bgr.shape[:2]
        W, H = self.yolo.in_w, self.yolo.in_h
        sw, sh = max(8, int(round(w * k))), max(8, int(round(h * k)))
        small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        fill = np.median(img_bgr[::8, ::8].reshape(-1, 3), axis=0).astype(img_bgr.dtype)

        def starts(size, win):
            if size <= win:
                return [0]
            n = int(np.ceil((size - win) / (win * (1 - self.tile_overlap)))) + 1
            return [int(round(i * (size - win) / (n - 1))) for i in range(n)]
        boxes, scores, dets, ms = [], [], [], 0.0
        for ty in starts(sh, H):
            for tx in starts(sw, W):
                tile = np.empty((H, W, 3), img_bgr.dtype)
                tile[:] = fill
                part = small[ty:ty + H, tx:tx + W]
                ox, oy = (W - part.shape[1]) // 2, (H - part.shape[0]) // 2   # centred if padded
                tile[oy:oy + part.shape[0], ox:ox + part.shape[1]] = part
                tdets, tms = self.yolo.predict(tile)
                ms += tms
                for d in tdets:
                    x1, y1, x2, y2 = d["box"]
                    x1, x2 = max(x1, ox), min(x2, ox + part.shape[1])
                    y1, y2 = max(y1, oy), min(y2, oy + part.shape[0])
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    b = [(x1 - ox + tx) / k, (y1 - oy + ty) / k, (x2 - ox + tx) / k, (y2 - oy + ty) / k]
                    d["box"] = [int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))]
                    dets.append(d)
                    boxes.append([d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]])
                    scores.append(float(d["confidence"]))
        if len(dets) > 1:                       # the same mine seen in two overlapping tiles
            keep = cv2.dnn.NMSBoxes(boxes, scores, 0.0, 0.3)
            keep = sorted((int(np.ravel(i)[0]) for i in keep), key=lambda i: -scores[i])
            out = []
            for i in keep:
                a = dets[i]["box"]
                if all(_contained(a, o["box"]) < 0.6 and _contained(o["box"], a) < 0.6 for o in out):
                    out.append(dets[i])
            dets = out
        return dets, ms

    def detect(self, img_bgr, px_per_m=None, height=None):
        """Raw detections (pixel boxes) and the total detector time in ms. px_per_m: ground
        scale of this frame (focal length / camera height), if known; frames far from the
        model's ref_px_per_m are rescaled (see _yolo_rescaled)."""
        dets, ms = [], 0.0
        if self.yolo is not None:
            h, w = img_bgr.shape[:2]
            fed = px_per_m * min(self.yolo.in_w / w, self.yolo.in_h / h) if px_per_m else None
            # shrink when mines look 1.4x too big (the 1 m landing check); tile only when they
            # are tiny (< 0.4x, a high-resolution camera fed whole): in between the model copes
            if fed and self.ref_px_per_m and not 0.4 < fed / self.ref_px_per_m < 1.4:
                dets, ms = self._yolo_rescaled(img_bgr, self.ref_px_per_m / px_per_m)
            else:
                dets, ms = self.yolo.predict(img_bgr)
            for d in dets:
                d.setdefault("source", "yolo")
        if self.color is not None:
            cdets, cms = self.color.predict(img_bgr, px_per_m)
            cdets = [d for d in cdets if d["confidence"] >= self.conf * 0.8]
            dets = merge_detections(dets, cdets) if self.yolo is not None else cdets
            ms += cms
        h, w = img_bgr.shape[:2]
        for d in dets:
            b = d["box"]
            d.setdefault("truncated", bool(b[0] <= 1 or b[1] <= 1 or b[2] >= w - 1 or b[3] >= h - 1))
        return dets, ms

    def locate(self, img_bgr, position, R_arena_body):
        """-> (detections, footprint (M, 2), detector ms). Each detection gets x, y (arena m),
        sigma (m), size (m, ground size of the box), valid (projects onto the ground and has
        a plausible size) and u, v (box centre pixel)."""
        h, w = img_bgr.shape[:2]
        cam = self.camera
        if (w, h) != (cam.width, cam.height):
            self.projector.camera = cam = cam.scaled_to(w, h)
        C = self.projector.camera_centre(position, R_arena_body)
        height = C[2] - self.projector.ground_z
        px_per_m = 0.5 * (cam.fx + cam.fy) / height if height > 0.2 else None
        dets, ms = self.detect(img_bgr, px_per_m, height if height > 0.2 else None)
        footprint = self.projector.footprint(position, R_arena_body) if height > 0.2 else np.zeros((0, 2))
        if dets:
            hits, size = self.projector.project_boxes([d["box"] for d in dets], position, R_arena_body)
            for d, p, ok, sg, sz in zip(dets, hits.points, hits.valid, hits.sigma, size):
                d["u"] = (d["box"][0] + d["box"][2]) / 2.0
                d["v"] = (d["box"][1] + d["box"][3]) / 2.0
                gsize = float(np.nanmax(sz)) if np.any(np.isfinite(sz)) else float("nan")
                d["size"] = gsize
                d["valid"] = bool(ok) and (d["truncated"] or self.min_size <= gsize <= self.max_size)
                d["x"], d["y"] = (float(p[0]), float(p[1])) if ok else (float("nan"), float("nan"))
                d["sigma"] = float(sg) * (self.truncated_sigma_scale if d["truncated"] else 1.0)
        return dets, footprint, ms


def draw_located(img_bgr, dets, map_overlay=None, hud=None):
    """Annotated copy: boxes with arena coordinates; map_overlay = [(u, v, label)] of known
    mines re-projected into this image; hud = list of text lines for the top banner."""
    out = img_bgr.copy()
    for u, v, label in map_overlay or []:
        if np.isfinite(u) and np.isfinite(v):
            cv2.circle(out, (int(u), int(v)), 16, (255, 0, 255), 1)
            cv2.putText(out, label, (int(u) + 14, int(v) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    for d in dets:
        x1, y1, x2, y2 = [int(a) for a in d["box"]]
        col = (0, 255, 0) if d["class_id"] == 0 else (0, 165, 255)
        if not d.get("valid", True):
            col = (128, 128, 128)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        if "x" in d and np.isfinite(d["x"]):
            label += f" ({d['x']:+.2f}, {d['y']:+.2f})"
        if d.get("mine_id"):
            label = f"#{d['mine_id']} " + label
        cv2.putText(out, label, (x1, max(34, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    if hud:
        cv2.rectangle(out, (0, 0), (out.shape[1], 14 * len(hud) + 8), (20, 20, 20), -1)
        for i, line in enumerate(hud):
            cv2.putText(out, line, (8, 15 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return out
