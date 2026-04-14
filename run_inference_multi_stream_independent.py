import os
import cv2
import time
import numpy as np
import threading
import torch
import torch.nn.functional as F
import maccel
from collections import defaultdict

# ========================
# Configuration
# ========================
# YOLOV8L_MXQ_PATH = "/home/mobilint/Desktop/yolov8l/yolov8l/yolov8l_global_v4.mxq"
YOLOV8_MXQ_PATH = "/home/mobilint/Desktop/yolov8l/model_quant/yolov8s_best_globalCore_v1.mxq"
IMG_SIZE = [640, 640]
NUM_CLASSES = 17
NUM_LAYERS = 3
REG_MAX = 16
CONF_THRESH = 0.5
IOU_THRESH = 0.

# Log cadence & warmup
PRINT_EVERY = int(os.getenv("PRINT_EVERY", 50))  # print stats every N frames per stream
WARMUP_FRAMES = int(os.getenv("WARMUP_FRAMES", 5))  # ignore first N frames in averages
EMA_ALPHA = float(os.getenv("EMA_ALPHA", 0.2))  # smoothing for overlay

# Stream list (placeholders)
RTSP_STREAMS = [
    "rtsp://admin:sd15553365!@223.171.50.201:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://shc:a12345678@shc000.iptime.org:554/cam/realmonitor?channel=2&subtype=0",
    "rtsp://admin:hlcctv1234!@112.222.237.114:40001/Streaming/Channels/101",
    "rtsp://admin:hlcctv1234!@112.222.237.114:40001/Streaming/Channels/501",
    "rtsp://admin:hlcctv1234!@112.222.237.114:40001/Streaming/Channels/601"
    # "rtsp://admin:hlcctv1234!@112.222.237.114:40001/Streaming/Unicast/channels/401"
] * 5

CLASS_NAMES = [
    "backhoe_loader", "cement_truck", "compactor", "dozer", "dump_truck",
    "excavator", "grader", "mobile_crane", "tower_crane", "wheel_loader",
    "worker", "Hardhat", "Red_Hardhat", "scaffolds", "Lifted Load",
    "Crane_Hook", "Hook"
]

device = torch.device("cpu")
stop_event = threading.Event()
frames_display = [None] * len(RTSP_STREAMS)
frame_locks = [threading.Lock() for _ in RTSP_STREAMS]
print(f"Total Stream: {len(RTSP_STREAMS)}")

# ========================
# Timing helpers
# ========================
def now():
    # perf_counter has better resolution & is monotonic
    return time.perf_counter()

class StreamStats:
    """Per-stream timing stats with running totals + EMA for HUD."""
    def __init__(self):
        self.frames = 0
        self._ema = {"pre": None, "infer": None, "post": None, "fps": None}

        # running sums for averages (skip warmup frames)
        self.sum_pre = 0.0
        self.sum_infer = 0.0
        self.sum_post = 0.0
        self.sum_total = 0.0
        self.count = 0

    def update(self, pre_s, infer_s, post_s, total_s):
        self.frames += 1

        # EMA for overlay
        def ema(name, val):
            if self._ema[name] is None:
                self._ema[name] = val
            else:
                self._ema[name] = self._ema[name] * (1 - EMA_ALPHA) + val * EMA_ALPHA
            return self._ema[name]

        ema_pre = ema("pre", pre_s)
        ema_infer = ema("infer", infer_s)
        ema_post = ema("post", post_s)
        ema_fps = ema("fps", (1.0 / total_s) if total_s > 0 else 0.0)

        # running sums after warmup
        if self.frames > WARMUP_FRAMES:
            self.sum_pre += pre_s
            self.sum_infer += infer_s
            self.sum_post += post_s
            self.sum_total += total_s
            self.count += 1

        return ema_pre, ema_infer, ema_post, ema_fps

    def should_print(self):
        return self.frames > 0 and (self.frames % PRINT_EVERY == 0)

    def averages_ms(self):
        if self.count == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        pre_ms = (self.sum_pre / self.count) * 1000.0
        infer_ms = (self.sum_infer / self.count) * 1000.0
        post_ms = (self.sum_post / self.count) * 1000.0
        total_ms = (self.sum_total / self.count) * 1000.0
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        return pre_ms, infer_ms, post_ms, total_ms, fps

# shared dict for stats per stream index
stats = defaultdict(StreamStats)
print_lock = threading.Lock()  # avoid interleaved prints

# ========================
# Core pipeline
# ========================
def preprocess_frame(frame):
    h0, w0 = frame.shape[:2]
    r = min(IMG_SIZE[0] / h0, IMG_SIZE[1] / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dw = IMG_SIZE[1] - new_unpad[0]
    dh = IMG_SIZE[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (frame.shape[1], frame.shape[0]) != new_unpad:
        frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    img = frame.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img[np.newaxis, :], frame  # NCHW + resized frame


def make_anchors(nl=3, img_size=IMG_SIZE, offset=0.5):
    imh, imw = img_size
    anchor_points, stride_tensor = [], []
    strides = [2 ** (3 + i) for i in range(nl)]
    for strd in strides:
        ny, nx = imh // strd, imw // strd
        sy = torch.arange(ny, dtype=torch.float32) + offset
        sx = torch.arange(nx, dtype=torch.float32) + offset
        yv, xv = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((xv, yv), -1).reshape(-1, 2))
        stride_tensor.append(torch.full((ny * nx, 1), strd, dtype=torch.float32))

    anchors = torch.cat(anchor_points, dim=0).T
    strides = torch.cat(stride_tensor, dim=0).T
    return anchors, strides


def decode_dfl(x, reg_max=REG_MAX):
    b, _, a = x.shape
    dfl_weight = torch.arange(reg_max, dtype=torch.float).reshape(1, -1, 1, 1)
    x = x.view(b, 4, reg_max, a).transpose(2, 1).softmax(1)
    return F.conv2d(x, dfl_weight).view(b, 4, a)


def postprocess(yolo_out, conf_thres, iou_thres):
    anchors, strides = make_anchors(NUM_LAYERS, IMG_SIZE)
    anchors, strides = anchors.to(device), strides.to(device)

    det_outs, cls_outs = [], []
    for tensor in yolo_out:
        t = torch.from_numpy(tensor).to(torch.float32) if not isinstance(tensor, torch.Tensor) else tensor
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.shape[1] == REG_MAX * 4:
            det_outs.append(t)
        elif t.shape[1] == NUM_CLASSES:
            cls_outs.append(t)

    det_outs = sorted(det_outs, key=lambda x: x.numel(), reverse=True)
    cls_outs = sorted(cls_outs, key=lambda x: x.numel(), reverse=True)

    outputs = [torch.cat((det, cls), dim=1).flatten(2) for det, cls in zip(det_outs, cls_outs)]
    batch_out = torch.cat(outputs, dim=2)[0]
    box_raw = batch_out[:REG_MAX * 4]
    cls_raw = batch_out[REG_MAX * 4:]

    scores = cls_raw.max(0)[0]
    keep = scores > -np.log(1 / conf_thres - 1)
    if keep.sum() == 0:
        return []

    box_raw, cls_raw = box_raw[:, keep], cls_raw[:, keep]
    anchors_keep, strides_keep = anchors[:, keep], strides[:, keep]
    dist = decode_dfl(box_raw.unsqueeze(0)).squeeze(0)
    boxes = dist2bbox(dist.T, anchors_keep.T, xywh=False) * strides_keep.T

    scores = cls_raw.sigmoid()
    conf, cls_idx = torch.max(scores, dim=0)
    dets = torch.cat([boxes, conf.unsqueeze(1), cls_idx.float().unsqueeze(1)], dim=1).cpu().numpy()

    return nms_numpy(dets, iou_thres, CONF_THRESH)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim) if xywh else torch.cat((x1y1, x2y2), dim)


def nms_numpy(dets, iou_thres=0.45, conf_thres=0.3):
    if len(dets) == 0:
        return []
    x1, y1, x2, y2, conf, cls = dets.T
    areas = (x2 - x1) * (y2 - y1)
    order = conf.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(dets[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
        if len(keep) >= 300:
            break
    return keep


def get_color(idx):
    np.random.seed(idx)
    return tuple(int(c) for c in np.random.choice(range(40, 256), size=3))


def draw_boxes(img, boxes, hud_text=None):
    for x1, y1, x2, y2, conf, cls_id in boxes:
        cls_id = int(cls_id)
        label = f"{CLASS_NAMES[cls_id]}: {conf*100:.1f}%"
        color = get_color(cls_id)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1 - 20)), (int(x1 + text_w + 4), int(y1)), color, -1)
        cv2.putText(img, label, (int(x1 + 2), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if hud_text:
        y = 24
        for line in hud_text.split("\n"):
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
            y += 26
    return img


# ========================
# Worker
# ========================
def stream_worker(idx, model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with print_lock:
            print(f"[Stream {idx}] failed to open: {video_path}")
        return

    while not stop_event.is_set():
        t_loop0 = now()

        ret, frame = cap.read()
        if not ret:
            continue

        t0 = now()
        inp, resized_frame = preprocess_frame(frame)
        t1 = now()

        # Inference
        output = model.infer(inp)  # assumes Python wrapper that accepts NCHW float array(s)
        t2 = now()

        # Postprocess (includes decode + NMS)
        detections = postprocess(output, CONF_THRESH, IOU_THRESH)
        t3 = now()

        # Timings
        pre_s = t1 - t0
        infer_s = t2 - t1
        post_s = t3 - t2
        total_s = t3 - t0
        ema_pre, ema_infer, ema_post, ema_fps = stats[idx].update(pre_s, infer_s, post_s, total_s)

        # Compose HUD text (EMA for visual stability)
        hud = (
            f"FPS: {ema_fps:5.2f}"
            f"  pre: {ema_pre*1000:6.1f} ms"
            f"  infer: {ema_infer*1000:6.1f} ms"
            f"  post: {ema_post*1000:6.1f} ms"
        )

        result = draw_boxes(resized_frame.copy(), detections, hud_text=hud)
        with frame_locks[idx]:
            frames_display[idx] = result

        # Periodic logging (running averages since warmup)
        if stats[idx].should_print():
            pre_ms, infer_ms, post_ms, total_ms, fps_avg = stats[idx].averages_ms()
            with print_lock:
                print(
                    f"[Stream {idx}] frames={stats[idx].frames} (avg after warmup)"
                    f" | pre={pre_ms:.2f} ms"
                    f" | infer={infer_ms:.2f} ms"
                    f" | post={post_ms:.2f} ms"
                    f" | total={total_ms:.2f} ms"
                    f" | FPS={fps_avg:.2f}"
                )

    cap.release()


# ========================
# Main
# ========================
def main():
    acc = maccel.Accelerator()
    config = maccel.ModelConfig()
    config.set_global8_core_mode()
    model = maccel.Model(YOLOV8_MXQ_PATH, config)
    model.launch(acc)

    threads = []
    for i, path in enumerate(RTSP_STREAMS):
        t = threading.Thread(target=stream_worker, args=(i, model, path), daemon=True)
        threads.append(t)
        t.start()

    # Mosaic layout
    cols = 6
    rows = 5
    screen_w = 1920
    screen_h = 1080
    tile_w = screen_w // cols
    tile_h = screen_h // rows
    win_w = tile_w * cols
    win_h = tile_h * rows

    cv2.namedWindow("Multi-Stream View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Stream View", win_w, win_h)

    try:
        while True:
            mosaic = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            for i in range(min(len(RTSP_STREAMS), rows * cols)):
                with frame_locks[i]:
                    frame = frames_display[i]
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                scale = min(tile_w / w, tile_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))

                canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                y_offset = (tile_h - new_h) // 2
                x_offset = (tile_w - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

                row, col = divmod(i, cols)
                y0, x0 = row * tile_h, col * tile_w
                mosaic[y0:y0+tile_h, x0:x0+tile_w] = canvas

            cv2.imshow("Multi-Stream View", mosaic)
            if cv2.waitKey(1) == 27:
                break

    finally:
        stop_event.set()
        for t in threads:
            t.join()
        model.dispose()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
