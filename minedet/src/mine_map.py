"""
Mine map: fuses projected detections from many frames (and drones) into a list of
mines in the arena frame, plus a coverage / occupancy grid with exclusion zones.

Per frame (MineMap.add_frame):
  1. detections closer than `frame_merge` (duplicate boxes, a peg's cap and base) merge
  2. each is matched one-to-one to the nearest track within `assoc_radius`, else it
     starts a new tentative track
  3. every track inside the frame's ground footprint gets a "view"; together with the
     hits this gives a detection ratio that exposes one-off false positives
Track position = weighted mean of its observations, weight = confidence / sigma^2
(sigma from GroundProjector: grows with range, off-nadir angle, cut-off boxes).
A track is a confirmed mine after `min_hits` hits in distinct frames, a detection
ratio of at least `min_ratio` and observations that agree within `max_spread`
(something above the ground, e.g. a pole top, drifts with parallax as the drone
moves). Confirmed mines get stable ids 1, 2, 3 ...

Grid (ROS OccupancyGrid convention): -1 unknown (never in a camera footprint),
0 seen and clear, `exclusion_value` within exclusion_radius of a mine's edge,
`mine_value` on the mine itself.
"""
import csv
import json
import math
import time

import cv2
import numpy as np

from geo_projection import point_in_polygon, shrink_polygon

CLASS_COLOURS = {"mine_disc": (40, 40, 230), "surface_marker": (0, 150, 255)}


class Track:
    __slots__ = ("key", "mine_id", "sw", "sx", "sy", "sxx", "syy", "hits", "views", "votes",
                 "conf_sum", "max_conf", "first_t", "last_t", "sources", "last_frame", "size_sum",
                 "hits_by", "views_by")

    def __init__(self, key):
        self.key, self.mine_id = key, None
        self.sw = self.sx = self.sy = self.sxx = self.syy = 0.0
        self.hits = self.views = 0
        self.votes = {}
        self.conf_sum = self.max_conf = self.size_sum = 0.0
        self.first_t = self.last_t = None
        self.sources = set()
        self.last_frame = None
        self.hits_by, self.views_by = {}, {}      # per source (drone)

    @property
    def x(self):
        return self.sx / self.sw

    @property
    def y(self):
        return self.sy / self.sw

    @property
    def spread(self):
        """Weighted standard deviation of the observations (m)."""
        vx = max(0.0, self.sxx / self.sw - self.x ** 2)
        vy = max(0.0, self.syy / self.sw - self.y ** 2)
        return math.sqrt(vx + vy)

    @property
    def cls(self):
        return max(self.votes, key=self.votes.get) if self.votes else "unknown"

    @property
    def score(self):
        return self.conf_sum / max(1, self.hits)

    @property
    def ratio(self):
        return self.hits / max(1, self.views, self.hits)

    def add(self, x, y, w, cls, conf, size, t, source, frame):
        self.sw += w
        self.sx += w * x
        self.sy += w * y
        self.sxx += w * x * x
        self.syy += w * y * y
        self.votes[cls] = self.votes.get(cls, 0.0) + conf
        if frame != self.last_frame:
            self.hits += 1
            self.hits_by[source] = self.hits_by.get(source, 0) + 1
            self.conf_sum += conf
            self.size_sum += size if np.isfinite(size) else 0.0
            self.last_frame = frame
        self.max_conf = max(self.max_conf, conf)
        self.first_t = t if self.first_t is None else self.first_t
        self.last_t = t
        self.sources.add(source)

    def absorb(self, other):
        for a in ("sw", "sx", "sy", "sxx", "syy", "conf_sum", "size_sum"):
            setattr(self, a, getattr(self, a) + getattr(other, a))
        self.hits += other.hits
        self.views = max(self.views, other.views)
        for k, v in other.votes.items():
            self.votes[k] = self.votes.get(k, 0.0) + v
        self.max_conf = max(self.max_conf, other.max_conf)
        self.first_t = min(t for t in (self.first_t, other.first_t) if t is not None)
        self.last_t = max(t for t in (self.last_t, other.last_t) if t is not None)
        self.sources |= other.sources
        for mine, theirs in ((self.hits_by, other.hits_by), (self.views_by, other.views_by)):
            for k, v in theirs.items():
                mine[k] = mine.get(k, 0) + v
        if self.mine_id is None or (other.mine_id is not None and other.mine_id < self.mine_id):
            self.mine_id = other.mine_id


class CoverageGrid:
    """Counts how often each cell was inside a camera footprint. Grows as needed."""

    def __init__(self, resolution=0.1, chunk=5.0):
        self.res = resolution
        self.chunk = chunk
        self.x0 = self.y0 = None           # arena coordinates of cell (0, 0)'s corner
        self.count = np.zeros((0, 0), dtype=np.uint16)

    def _ensure(self, xmin, ymin, xmax, ymax):
        c = self.chunk
        xmin, ymin = math.floor(xmin / c) * c, math.floor(ymin / c) * c
        xmax, ymax = math.ceil(xmax / c) * c, math.ceil(ymax / c) * c
        if self.x0 is None:
            self.x0, self.y0 = xmin, ymin
            self.count = np.zeros((round((ymax - ymin) / self.res), round((xmax - xmin) / self.res)), np.uint16)
            return
        h, w = self.count.shape
        ox, oy = min(self.x0, xmin), min(self.y0, ymin)
        ex, ey = max(self.x0 + w * self.res, xmax), max(self.y0 + h * self.res, ymax)
        if (ox, oy) == (self.x0, self.y0) and ex <= self.x0 + w * self.res + 1e-9 and ey <= self.y0 + h * self.res + 1e-9:
            return
        new = np.zeros((round((ey - oy) / self.res), round((ex - ox) / self.res)), np.uint16)
        i, j = round((self.y0 - oy) / self.res), round((self.x0 - ox) / self.res)
        new[i:i + h, j:j + w] = self.count
        self.count, self.x0, self.y0 = new, ox, oy

    def mark(self, polygon):
        poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(poly) < 3:
            return
        self._ensure(*(poly.min(axis=0) - self.res), *(poly.max(axis=0) + self.res))
        cells = np.round((poly - [self.x0, self.y0]) / self.res * 8).astype(np.int32)   # 3 fractional bits
        mask = np.zeros(self.count.shape, np.uint8)
        cv2.fillPoly(mask, [cells.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8, shift=3)
        self.count += mask

    @property
    def shape(self):
        return self.count.shape

    def cell_centres(self):
        h, w = self.count.shape
        return (self.x0 + (np.arange(w) + 0.5) * self.res, self.y0 + (np.arange(h) + 0.5) * self.res)


class MineMap:
    def __init__(self, assoc_radius=0.5, frame_merge=0.3, merge_radius=0.3, min_hits=3, min_ratio=0.25,
                 max_spread=0.2, drop_after_views=15, drop_ratio=0.1, footprint_margin=0.15,
                 mine_radius=0.15, exclusion_radius=1.0, grid_resolution=0.1,
                 mine_value=100, exclusion_value=100, field_tf=None):
        self.assoc_radius, self.frame_merge, self.merge_radius = assoc_radius, frame_merge, merge_radius
        self.min_hits, self.min_ratio, self.max_spread = min_hits, min_ratio, max_spread
        self.drop_after_views, self.drop_ratio = drop_after_views, drop_ratio
        self.footprint_margin = footprint_margin
        self.mine_radius, self.exclusion_radius = mine_radius, exclusion_radius
        self.mine_value, self.exclusion_value = mine_value, exclusion_value
        self.field_tf = field_tf               # (x0, y0, yaw): arena -> competition field frame
        self.grid = CoverageGrid(grid_resolution)
        self.tracks = []
        self.frames = 0
        self._next_key = 0
        self._next_mine = 1
        self.version = 0                       # bumps whenever the confirmed map changes

    # ------------------------------------------------------------------ updates
    def add_frame(self, detections, footprint=None, t=None, source=""):
        """detections: iterable of dicts {x, y, sigma, class_name, confidence[, size]} (arena m).
        footprint: (M, 2) ground outline of the image, or None. Returns the track key per detection."""
        t = time.time() if t is None else t
        self.frames += 1
        frame = (source, self.frames)
        if footprint is not None and len(footprint) >= 3:
            self.grid.mark(footprint)
            inner = shrink_polygon(footprint, self.footprint_margin)
            if self.tracks:
                inside = point_in_polygon([(tr.x, tr.y) for tr in self.tracks], inner)
                for tr, ok in zip(self.tracks, inside):
                    if ok:
                        tr.views += 1
                        tr.views_by[source] = tr.views_by.get(source, 0) + 1

        dets = self._merge_in_frame([d for d in detections if np.isfinite(d["x"]) and np.isfinite(d["y"])])
        keys = [None] * len(dets)
        pairs = []
        for i, d in enumerate(dets):
            for tr in self.tracks:
                dist = math.hypot(d["x"] - tr.x, d["y"] - tr.y)
                if dist < self.assoc_radius:
                    pairs.append((dist, i, tr))
        used = set()
        for dist, i, tr in sorted(pairs, key=lambda p: p[0]):
            if keys[i] is None and tr.key not in used:
                self._add(tr, dets[i], t, source, frame)
                keys[i] = tr.key
                used.add(tr.key)
        for i, d in enumerate(dets):
            if keys[i] is None:
                tr = Track(self._next_key)
                self._next_key += 1
                tr.views = 1 if footprint is not None else 0
                if footprint is not None:
                    tr.views_by[source] = 1
                self._add(tr, d, t, source, frame)
                self.tracks.append(tr)
                keys[i] = tr.key
        self._merge_tracks()
        self._prune()
        self._confirm()
        return keys

    def _add(self, tr, d, t, source, frame):
        sigma = max(float(d.get("sigma", 0.05)), 0.01)
        conf = float(d.get("confidence", 1.0))
        tr.add(d["x"], d["y"], conf / sigma ** 2, d.get("class_name", "mine_disc"), conf,
               float(d.get("size", float("nan"))), t, source, frame)

    def _merge_in_frame(self, dets):
        """One observation per object per frame: merge detections closer than frame_merge."""
        out = []
        for d in sorted(dets, key=lambda d: -d.get("confidence", 1.0)):
            for o in out:
                if math.hypot(d["x"] - o["x"], d["y"] - o["y"]) < self.frame_merge:
                    wa = o["confidence"] / o["sigma"] ** 2
                    wb = d.get("confidence", 1.0) / max(d.get("sigma", 0.05), 0.01) ** 2
                    o["x"] = (wa * o["x"] + wb * d["x"]) / (wa + wb)
                    o["y"] = (wa * o["y"] + wb * d["y"]) / (wa + wb)
                    o["sigma"] = 1.0 / math.sqrt((wa + wb) / o["confidence"])
                    break
            else:
                o = dict(d)
                o.setdefault("confidence", 1.0)
                o["sigma"] = max(float(o.get("sigma", 0.05)), 0.01)
                out.append(o)
        return out

    def _merge_tracks(self):
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(self.tracks):
                for b in self.tracks[i + 1:]:
                    if math.hypot(a.x - b.x, a.y - b.y) < self.merge_radius:
                        keep, gone = (a, b) if a.sw >= b.sw else (b, a)
                        keep.absorb(gone)
                        self.tracks.remove(gone)
                        self.version += gone.mine_id is not None
                        changed = True
                        break
                if changed:
                    break

    def _prune(self):
        """Forget tentative tracks that were in view many times but hardly ever detected."""
        keep = []
        for tr in self.tracks:
            if tr.mine_id is None and tr.views >= self.drop_after_views and tr.ratio < self.drop_ratio:
                continue
            keep.append(tr)
        self.tracks = keep

    def _confirm(self):
        for tr in self.tracks:
            if (tr.mine_id is None and tr.hits >= self.min_hits and tr.ratio >= self.min_ratio
                    and tr.spread <= self.max_spread):
                tr.mine_id = self._next_mine
                self._next_mine += 1
                self.version += 1

    # ------------------------------------------------------------------ outputs
    def to_field(self, x, y):
        if self.field_tf is None:
            return None
        x0, y0, yaw = self.field_tf
        c, s = math.cos(yaw), math.sin(yaw)
        return x0 + c * x - s * y, y0 + s * x + c * y

    def mines(self, confirmed_only=True):
        out = []
        for tr in sorted(self.tracks, key=lambda t: (t.mine_id is None, t.mine_id or 0, t.key)):
            if confirmed_only and tr.mine_id is None:
                continue
            m = {"id": tr.mine_id, "x": round(tr.x, 3), "y": round(tr.y, 3), "class": tr.cls,
                 "score": round(tr.score, 3), "max_conf": round(tr.max_conf, 3), "hits": tr.hits,
                 "views": tr.views, "ratio": round(tr.ratio, 2), "spread": round(tr.spread, 3),
                 "size": round(tr.size_sum / max(1, tr.hits), 3),
                 "sources": sorted(tr.sources), "hits_by": dict(tr.hits_by), "views_by": dict(tr.views_by),
                 "first_seen": tr.first_t, "last_seen": tr.last_t,
                 "confirmed": tr.mine_id is not None}
            f = self.to_field(tr.x, tr.y)
            if f is not None:
                m["field_x"], m["field_y"] = round(f[0], 3), round(f[1], 3)
            out.append(m)
        return out

    def coverage_area(self):
        return float(np.count_nonzero(self.grid.count)) * self.grid.res ** 2

    def to_dict(self):
        return {"frame": "arena", "exclusion_radius": self.exclusion_radius, "mine_radius": self.mine_radius,
                "field_tf": self.field_tf, "frames": self.frames, "coverage_m2": round(self.coverage_area(), 2),
                "mines": self.mines(True), "tentative": self.mines(False)[len(self.mines(True)):]}

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    def save_csv(self, path):
        cols = ["id", "x", "y", "field_x", "field_y", "class", "score", "max_conf", "hits", "views",
                "ratio", "spread", "size"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for m in self.mines(True):
                w.writerow(m)

    def occupancy(self):
        """-> (int8 grid [rows = y, cols = x], origin x, origin y, resolution): ROS OccupancyGrid layout."""
        g = self.grid
        if g.x0 is None:
            return np.full((1, 1), -1, np.int8), 0.0, 0.0, g.res
        occ = np.where(g.count > 0, 0, -1).astype(np.int8)
        xs, ys = g.cell_centres()
        for m in self.mines(True):
            ix = slice(max(0, np.searchsorted(xs, m["x"] - 2)), np.searchsorted(xs, m["x"] + 2))
            iy = slice(max(0, np.searchsorted(ys, m["y"] - 2)), np.searchsorted(ys, m["y"] + 2))
            X, Y = np.meshgrid(xs[ix], ys[iy])
            d = np.hypot(X - m["x"], Y - m["y"])
            sub = occ[iy, ix]
            sub[d <= self.mine_radius + self.exclusion_radius] = np.maximum(
                sub[d <= self.mine_radius + self.exclusion_radius], self.exclusion_value)
            sub[d <= self.mine_radius] = self.mine_value
        return occ, g.x0, g.y0, g.res

    def render(self, max_width=1400, px_per_m=40.0, drones=None, truth=None, title=None):
        """Top-down picture of the map (arena x right, y up). drones: {name: (x, y, z)};
        truth: [(x, y)] ground truth (sim) drawn as small crosses."""
        g = self.grid
        pts = [(tr.x, tr.y) for tr in self.tracks] + [p[:2] for p in (drones or {}).values()] + list(truth or [])
        if g.x0 is not None:
            h, w = g.shape
            pts += [(g.x0, g.y0), (g.x0 + w * g.res, g.y0 + h * g.res)]
        if not pts:
            pts = [(-1.0, -1.0), (1.0, 1.0)]
        P = np.array(pts, dtype=float)
        xmin, ymin = P.min(axis=0) - 1.5
        xmax, ymax = P.max(axis=0) + 1.5
        s = min(px_per_m, max_width / (xmax - xmin))
        W, H = int((xmax - xmin) * s), int((ymax - ymin) * s) + 30

        def px(x, y):
            return int(round((x - xmin) * s)), int(round(H - (y - ymin) * s))

        img = np.full((H, W, 3), 35, np.uint8)
        if g.x0 is not None:                   # seen area
            cov = (g.count > 0).astype(np.uint8)[::-1]
            x0p, y1p = px(g.x0, g.y0 + g.shape[0] * g.res)
            big = cv2.resize(cov, (max(1, int(round(g.shape[1] * g.res * s))), max(1, int(round(g.shape[0] * g.res * s)))),
                             interpolation=cv2.INTER_NEAREST)
            ys, xs = slice(max(0, y1p), min(H, y1p + big.shape[0])), slice(max(0, x0p), min(W, x0p + big.shape[1]))
            region = img[ys, xs]
            m = big[:region.shape[0], :region.shape[1]].astype(bool)
            region[m] = (70, 95, 70)
        for gx in range(math.ceil(xmin), math.floor(xmax) + 1):   # 1 m grid
            cv2.line(img, px(gx, ymin), px(gx, ymax), (55, 55, 55) if gx else (120, 120, 120), 1)
        for gy in range(math.ceil(ymin), math.floor(ymax) + 1):
            cv2.line(img, px(xmin, gy), px(xmax, gy), (55, 55, 55) if gy else (120, 120, 120), 1)
        overlay = img.copy()
        for m in self.mines(True):
            cv2.circle(overlay, px(m["x"], m["y"]), int((self.exclusion_radius + self.mine_radius) * s), (30, 30, 150), -1)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        for tx, ty in truth or []:
            c = px(tx, ty)
            cv2.drawMarker(img, c, (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
        for tr in self.tracks:
            c = px(tr.x, tr.y)
            if tr.mine_id is None:
                cv2.circle(img, c, max(3, int(self.mine_radius * s)), (150, 150, 150), 1)
                continue
            col = CLASS_COLOURS.get(tr.cls, (200, 200, 200))
            cv2.circle(img, c, max(4, int(self.mine_radius * s)), col, -1)
            cv2.putText(img, f"#{tr.mine_id}", (c[0] + 7, c[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for name, p in (drones or {}).items():
            c = px(p[0], p[1])
            cv2.drawMarker(img, c, (255, 220, 0), cv2.MARKER_TRIANGLE_UP, 14, 2)
            cv2.putText(img, name, (c[0] + 8, c[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 0), 1)
        n = len(self.mines(True))
        text = title or f"mines {n} (tentative {len(self.tracks) - n}) | seen {self.coverage_area():.1f} m2 | 1 m grid, arena frame"
        cv2.rectangle(img, (0, 0), (W, 24), (20, 20, 20), -1)
        cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        return img
