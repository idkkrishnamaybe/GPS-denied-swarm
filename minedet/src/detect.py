import os
import sys
import time
import glob
import json
import zipfile
import argparse
import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            import tensorflow.lite as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            Interpreter = None      # reported when a model is loaded (the colour detector still works)

NO_RUNTIME = ("Neither ai_edge_litert, tflite_runtime, nor tensorflow.lite is installed. "
              "Run: pip install ai-edge-litert (keep numpy<2 next to ROS: pip install ai-edge-litert 'numpy<2')")

MODEL_PRESETS = {
    "640x384": "models/yolo26n_640x384_int8.tflite",
    "640x480": "models/yolo26n_640x480_int8.tflite",
    "fast": "models/yolo26n_640x384_int8.tflite",
    "full": "models/yolo26n_640x480_int8.tflite"
}

def read_class_names(model_path):
    """Class names from the Ultralytics metadata embedded in the .tflite (a zip appendix)."""
    try:
        with zipfile.ZipFile(model_path) as z:
            names = json.loads(z.read("metadata.json")).get("names", {})
        return {int(k): str(v) for k, v in names.items()} or None
    except Exception:
        return None


def resolve_model_path(model_path):
    if model_path in MODEL_PRESETS:
        model_path = MODEL_PRESETS[model_path]
    if os.path.exists(model_path):
        return os.path.abspath(model_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alt_path = os.path.join(repo_root, model_path)
    if os.path.exists(alt_path):
        return os.path.abspath(alt_path)
    raise FileNotFoundError(f"Model file not found: {model_path}")

def contained(a, b):
    """Fraction of box a's area inside box b ([x1, y1, x2, y2])."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


class YOLO26LiteRT:
    def __init__(self, model_path="models/yolo26n_640x384_int8.tflite", threads=2, conf_thresh=0.35, iou_thresh=0.45,
                 contain_thresh=0.7):
        if Interpreter is None:
            raise ImportError(NO_RUNTIME)
        self.model_path = resolve_model_path(model_path)
        self.threads = threads
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # a box lying mostly inside a stronger one is a duplicate (plain IoU NMS keeps it)
        self.contain_thresh = contain_thresh
        self.class_names = read_class_names(self.model_path) or {0: "mine_disc", 1: "surface_marker"}

        # Initialize LiteRT Interpreter with XNNPACK Delegate
        self.interpreter = Interpreter(model_path=self.model_path, num_threads=self.threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.in_shape = self.input_details[0]["shape"]
        self.in_dtype = self.input_details[0]["dtype"]
        self.in_idx = self.input_details[0]["index"]
        self.out_idx = self.output_details[0]["index"]

        self.input_quant = self.input_details[0].get("quantization_parameters", {})
        self.output_quant = self.output_details[0].get("quantization_parameters", {})
        
        in_scales = self.input_quant.get("scales")
        in_zeros = self.input_quant.get("zero_points")
        self.in_scale = float(in_scales[0]) if in_scales is not None and len(in_scales) > 0 else 1.0
        self.in_zero = int(in_zeros[0]) if in_zeros is not None and len(in_zeros) > 0 else 0
        if self.in_scale == 0:
            self.in_scale = 1.0 / 255.0

        out_scales = self.output_quant.get("scales")
        out_zeros = self.output_quant.get("zero_points")
        self.out_scale = float(out_scales[0]) if out_scales is not None and len(out_scales) > 0 else 1.0
        self.out_zero = float(out_zeros[0]) if out_zeros is not None and len(out_zeros) > 0 else 0.0

        if len(self.in_shape) == 4:
            if self.in_shape[1] == 3:  # NCHW
                self.in_h, self.in_w = int(self.in_shape[2]), int(self.in_shape[3])
                self.is_nchw = True
            else:  # NHWC
                self.in_h, self.in_w = int(self.in_shape[1]), int(self.in_shape[2])
                self.is_nchw = False

    def preprocess(self, img_bgr):
        h_orig, w_orig = img_bgr.shape[:2]
        
        # Zero-copy if already matching input dimensions (from Picamera2 ISP)
        if h_orig == self.in_h and w_orig == self.in_w:
            resized = img_bgr
        else:
            resized = cv2.resize(img_bgr, (self.in_w, self.in_h))
            
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.in_dtype == np.int8:
            if abs(self.in_scale - (1.0 / 255.0)) < 1e-4 and self.in_zero == -128:
                input_data = (rgb.astype(np.int16) - 128).astype(np.int8)
            else:
                norm = (rgb.astype(np.float32) / 255.0) / self.in_scale + self.in_zero
                input_data = np.clip(norm, -128, 127).astype(np.int8)
        elif self.in_dtype == np.uint8:
            if abs(self.in_scale - (1.0 / 255.0)) < 1e-4 and self.in_zero == 0:
                input_data = rgb
            else:
                norm = (rgb.astype(np.float32) / 255.0) / self.in_scale + self.in_zero
                input_data = np.clip(norm, 0, 255).astype(np.uint8)
        else:
            input_data = (rgb.astype(np.float32) / 255.0).astype(np.float32)

        if self.is_nchw:
            input_data = np.transpose(input_data, (2, 0, 1))

        input_tensor = np.expand_dims(input_data, axis=0)
        return input_tensor, (h_orig, w_orig)

    def predict(self, img_bgr):
        input_tensor, (h_orig, w_orig) = self.preprocess(img_bgr)

        self.interpreter.set_tensor(self.in_idx, input_tensor)
        t0 = time.perf_counter()
        self.interpreter.invoke()
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        raw_output = self.interpreter.get_tensor(self.out_idx)

        # Fast dequantize output if quantized
        if self.output_details[0]["dtype"] in (np.int8, np.uint8):
            raw_output = (raw_output.astype(np.float32) - self.out_zero) * self.out_scale

        detections = []
        if len(raw_output.shape) == 3:
            if raw_output.shape[1] < raw_output.shape[2]:  # [1, 6, num_boxes]
                preds = raw_output[0].T
            else:  # [1, num_boxes, 6]
                preds = raw_output[0]

            # Vectorized confidence score & class extraction (NumPy C-speed)
            class_scores = preds[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(len(class_scores)), class_ids]

            mask = scores >= self.conf_thresh
            if np.any(mask):
                valid_preds = preds[mask]
                valid_scores = scores[mask]
                valid_cls_ids = class_ids[mask]

                cx = valid_preds[:, 0]
                cy = valid_preds[:, 1]
                w = valid_preds[:, 2]
                h = valid_preds[:, 3]

                x1 = ((cx - w / 2.0) * w_orig).astype(int)
                y1 = ((cy - h / 2.0) * h_orig).astype(int)
                wb = (w * w_orig).astype(int)
                hb = (h * h_orig).astype(int)

                boxes = np.stack([x1, y1, wb, hb], axis=1).tolist()
                scores_list = valid_scores.tolist()

                indices = cv2.dnn.NMSBoxes(boxes, scores_list, self.conf_thresh, self.iou_thresh)
                if len(indices) > 0:
                    for i in indices:
                        idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else int(i)
                        x, y, box_w, box_h = boxes[idx]
                        cls_id = int(valid_cls_ids[idx])
                        score = float(scores_list[idx])
                        cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                        box = [max(0, x), max(0, y), min(w_orig, x + box_w), min(h_orig, y + box_h)]
                        if any(contained(box, d["box"]) > self.contain_thresh for d in detections):
                            continue        # NMSBoxes returns strongest first
                        detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": score,
                            "box": box,
                            # cut by the image border: the box centre is not the object's centre
                            "truncated": bool(box[0] <= 1 or box[1] <= 1 or box[2] >= w_orig - 1 or box[3] >= h_orig - 1)
                        })

        return detections, latency_ms

    def draw_detections(self, img_bgr, detections):
        annotated = img_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cls_name = d["class_name"]
            score = d["confidence"]

            color = (0, 255, 0) if d["class_id"] == 0 else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{cls_name} {score:.2f}", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated

def main():
    parser = argparse.ArgumentParser(description="YOLO26 LiteRT Inference on Raspberry Pi")
    parser.add_argument("--model", default="640x384", help="Model preset ('640x384', '640x480', 'fast', 'full') or file path")
    parser.add_argument("--source", default="test_images", help="Image path or folder")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--threads", type=int, default=2, help="Number of CPU threads")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    detector = YOLO26LiteRT(model_path=args.model, threads=args.threads, conf_thresh=args.conf)
    os.makedirs(args.out, exist_ok=True)

    source_path = args.source
    if not os.path.exists(source_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_source = os.path.join(repo_root, args.source)
        if os.path.exists(alt_source):
            source_path = alt_source

    if os.path.isdir(source_path):
        images = sorted(glob.glob(os.path.join(source_path, "*.jpg")) + glob.glob(os.path.join(source_path, "*.png")))
    elif os.path.isfile(source_path):
        images = [source_path]
    else:
        print(f"[ERROR] Source not found: {args.source}")
        sys.exit(1)

    print(f"Running detection on {len(images)} images")
    print(f"Model: {detector.model_path} ({detector.in_w}x{detector.in_h}) | Threads: {args.threads}\n")
    
    total_latencies = []
    total_discs = 0
    total_markers = 0
    total_targets = 0

    for img_p in images:
        img = cv2.imread(img_p)
        if img is None:
            continue
        fname = os.path.basename(img_p)
        dets, lat_ms = detector.predict(img)
        total_latencies.append(lat_ms)

        annotated = detector.draw_detections(img, dets)
        out_path = os.path.join(args.out, f"detected_{fname}")
        cv2.imwrite(out_path, annotated)

        disc_count = sum(1 for d in dets if d["class_id"] == 0)
        marker_count = sum(1 for d in dets if d["class_id"] == 1)
        total_discs += disc_count
        total_markers += marker_count
        total_targets += len(dets)

        print(f" [{fname}] -> {len(dets)} targets (discs: {disc_count}, markers: {marker_count}) | Latency: {lat_ms:.2f} ms")

    if total_latencies:
        mean_lat = float(np.mean(total_latencies))
        std_lat = float(np.std(total_latencies))
        p50_lat = float(np.percentile(total_latencies, 50))
        p95_lat = float(np.percentile(total_latencies, 95))
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        print(f"\n========================================================================")
        print(f" YOLO26n LiteRT INFERENCE SUMMARY TABLE")
        print(f"========================================================================")
        print(f" Images Processed     : {len(total_latencies)}")
        print(f" Model Path           : {os.path.basename(detector.model_path)} ({detector.in_w}x{detector.in_h})")
        print(f" CPU Threads          : {args.threads} Cores")
        print(f" Confidence Threshold : {args.conf}")
        print(f"------------------------------------------------------------------------")
        print(f" TARGET CLASS COUNTS:")
        print(f"  • mine_disc        (cls 0) : {total_discs}")
        print(f"  • surface_marker   (cls 1) : {total_markers}")
        print(f"  • Total Detections         : {total_targets}")
        print(f"------------------------------------------------------------------------")
        print(f" LATENCY & THROUGHPUT METRICS:")
        print(f"  • Mean Latency             : {mean_lat:.2f} ± {std_lat:.2f} ms")
        print(f"  • Median Latency (p50)     : {p50_lat:.2f} ms")
        print(f"  • 95th Percentile (p95)    : {p95_lat:.2f} ms")
        print(f"  • Throughput               : {fps:.2f} FPS")
        print(f"------------------------------------------------------------------------")
        print(f" Results Saved In     : {args.out}/")
        print(f"========================================================================\n")

if __name__ == "__main__":
    main()
