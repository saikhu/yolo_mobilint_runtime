#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import time
import json
import math
import queue
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os, glob
from pathlib import Path
import numpy as np
import cv2
import psutil, subprocess
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt



# ---------- Defaults (edit to your environment) ----------
DEFAULT_CLASS_NAMES = [
    "backhoe_loader", "cement_truck", "compactor", "dozer", "dump_truck",
    "excavator", "grader", "mobile_crane", "tower_crane", "wheel_loader",
    "worker", "Hardhat", "Red_Hardhat", "scaffolds", "Lifted Load",
    "Crane_Hook", "Hook"
]

# Sensible defaults; override via CLI
DEFAULT_NPU_MXQ = "../models/yolov8s_best_globalCore_v1.mxq"
DEFAULT_GPU_TORCHSCRIPT = "../models/yolov8s_best.pt"  
DEFAULT_STREAMS = ["0.mp4"]  # or add RTSP URLs here


# ---------- Safe import of engines ----------
try:
    from yolo_mobilint_runtime.src.npu_inference_yolov8 import NPUInferenceEngine, InferenceConfig
except Exception as e:
    NPUInferenceEngine = None
    InferenceConfig = None

try:
    from gpu_yolov8_inference import GPUYOLOv8Engine
except Exception as e:
    GPUYOLOv8Engine = None


# ---------- Utility ----------
def now() -> float:
    return time.perf_counter()


def percentiles(arr: List[float], pcts=(50, 90, 99)) -> Dict[str, float]:
    if not arr:
        return {f"p{p}": float("nan") for p in pcts}
    a = np.asarray(arr, dtype=np.float64)
    out = {}
    for p in pcts:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out

def plot_compare_latency(prefix: str, gpu_dev: "DeviceSummary", npu_dev: "DeviceSummary", out_dir: str = "multistrean_benchmark_20s"):
    os.makedirs(out_dir, exist_ok=True)
    labels = ["mean", "p50", "p90", "p99"]
    gpu_vals = [gpu_dev.lat_total_mean_ms, gpu_dev.lat_total_p50_ms, gpu_dev.lat_total_p90_ms, gpu_dev.lat_total_p99_ms]
    npu_vals = [npu_dev.lat_total_mean_ms, npu_dev.lat_total_p50_ms, npu_dev.lat_total_p90_ms, npu_dev.lat_total_p99_ms]
    x = np.arange(len(labels)); w = 0.35
    plt.figure()
    plt.bar(x - w/2, gpu_vals, width=w, label="GPU")
    plt.bar(x + w/2, npu_vals, width=w, label="NPU")
    plt.xticks(x, labels); plt.ylabel("Latency (ms)"); plt.title("Latency summary: GPU vs NPU"); plt.legend()
    path = os.path.join(out_dir, f"{prefix}_compare_latency.png")
    plt.tight_layout(); plt.savefig(path); plt.close(); return path

def plot_compare_overall_fps(prefix: str, gpu_dev: "DeviceSummary", npu_dev: "DeviceSummary", out_dir: str = "multistrean_benchmark_20s"):
    os.makedirs(out_dir, exist_ok=True)
    labels = ["GPU", "NPU"]; vals = [gpu_dev.overall_fps, npu_dev.overall_fps]
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Overall FPS"); plt.title("Overall FPS: GPU vs NPU")
    path = os.path.join(out_dir, f"{prefix}_compare_overall_fps.png")
    plt.tight_layout(); plt.savefig(path); plt.close(); return path

def plot_compare_stream_fps(prefix: str, gpu_summ: List["StreamMetrics"], npu_summ: List["StreamMetrics"], out_dir: str = "multistrean_benchmark_20s"):
    os.makedirs(out_dir, exist_ok=True)
    # align by stream_id present in both (intersection)
    g_map = {m.stream_id: m.summary()["fps_processed"] for m in gpu_summ}
    n_map = {m.stream_id: m.summary()["fps_processed"] for m in npu_summ}
    ids = sorted(set(g_map) & set(n_map))
    if not ids: return None
    g_vals = [g_map[i] for i in ids]
    n_vals = [n_map[i] for i in ids]
    x = np.arange(len(ids)); w = 0.35
    plt.figure()
    plt.bar(x - w/2, g_vals, width=w, label="GPU")
    plt.bar(x + w/2, n_vals, width=w, label="NPU")
    plt.xticks(x, [str(i) for i in ids]); plt.xlabel("Stream ID"); plt.ylabel("FPS (processed)")
    plt.title("Per-stream FPS: GPU vs NPU"); plt.legend()
    path = os.path.join(out_dir, f"{prefix}_compare_stream_fps.png")
    plt.tight_layout(); plt.savefig(path); plt.close(); return path

def _norm_time(rows: List[Dict[str, Any]]):
    if not rows: return [], {}
    t0 = rows[0]["ts"]; t = [r["ts"] - t0 for r in rows]
    return t, rows

def plot_compare_resources(prefix: str, gpu_rows: List[Dict[str, Any]], npu_rows: List[Dict[str, Any]], out_dir: str = "multistrean_benchmark_20s"):
    if not (gpu_rows or npu_rows): return []
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    g_t, _ = _norm_time(gpu_rows); n_t, _ = _norm_time(npu_rows)

    def line(x, y, label):
        plt.plot(x, y, label=label)

    # CPU total %
    plt.figure()
    if gpu_rows: line(g_t, [r.get("cpu_total_pct", np.nan) for r in gpu_rows], "GPU")
    if npu_rows: line(n_t, [r.get("cpu_total_pct", np.nan) for r in npu_rows], "NPU")
    plt.xlabel("Time (s)"); plt.ylabel("CPU total (%)"); plt.title("CPU total: GPU vs NPU"); plt.legend()
    p = os.path.join(out_dir, f"{prefix}_compare_res_cpu_total.png"); plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    # CPU proc %
    plt.figure()
    if gpu_rows: line(g_t, [r.get("cpu_proc_pct", np.nan) for r in gpu_rows], "GPU")
    if npu_rows: line(n_t, [r.get("cpu_proc_pct", np.nan) for r in npu_rows], "NPU")
    plt.xlabel("Time (s)"); plt.ylabel("CPU process (%)"); plt.title("CPU process: GPU vs NPU"); plt.legend()
    p = os.path.join(out_dir, f"{prefix}_compare_res_cpu_proc.png"); plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    # RSS MB
    plt.figure()
    if gpu_rows: line(g_t, [r.get("mem_proc_rss_mb", np.nan) for r in gpu_rows], "GPU")
    if npu_rows: line(n_t, [r.get("mem_proc_rss_mb", np.nan) for r in npu_rows], "NPU")
    plt.xlabel("Time (s)"); plt.ylabel("RSS (MB)"); plt.title("Process RSS: GPU vs NPU"); plt.legend()
    p = os.path.join(out_dir, f"{prefix}_compare_res_rss.png"); plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    # (GPU-only) util & mem used — still useful to see alongside NPU CPU
    if gpu_rows and any("gpu_util_pct" in r for r in gpu_rows):
        plt.figure()
        line(g_t, [r.get("gpu_util_pct", np.nan) for r in gpu_rows], "GPU")
        plt.xlabel("Time (s)"); plt.ylabel("GPU util (%)"); plt.title("GPU util"); plt.legend()
        p = os.path.join(out_dir, f"{prefix}_compare_res_gpu_util.png"); plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    if gpu_rows and any("gpu_mem_used_mb" in r for r in gpu_rows):
        plt.figure()
        line(g_t, [r.get("gpu_mem_used_mb", np.nan) for r in gpu_rows], "GPU")
        plt.xlabel("Time (s)"); plt.ylabel("GPU mem used (MB)"); plt.title("GPU memory used"); plt.legend()
        p = os.path.join(out_dir, f"{prefix}_compare_res_gpu_mem.png"); plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    return saved

class ResourceMonitor(threading.Thread):
    def __init__(self, device_label: str, interval: float = 0.5, with_gpu: bool = False):
        super().__init__(daemon=True)
        self.device_label = device_label
        self.interval = interval
        self.with_gpu = with_gpu
        self.stop_event = threading.Event()
        self.rows: List[Dict[str, Any]] = []
        self.proc = psutil.Process(os.getpid())

    def run(self):
        # Prime CPU percent for process
        self.proc.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            row = self.sample_once()
            self.rows.append(row)
            time.sleep(self.interval)

    def sample_once(self) -> Dict[str, Any]:
        ts = time.time()
        vm = psutil.virtual_memory()
        row = {
            "ts": ts,
            "cpu_total_pct": psutil.cpu_percent(interval=None),
            "cpu_proc_pct": self.proc.cpu_percent(interval=None),
            "mem_proc_rss_mb": self.proc.memory_info().rss / (1024*1024),
            "mem_sys_used_mb": (vm.total - vm.available) / (1024*1024),
        }
        if self.with_gpu:
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ], stderr=subprocess.DEVNULL, timeout=0.3).decode().strip()
                # Handle multi-GPU: we take the first line (GPU 0) by default
                line = out.splitlines()[0]
                util, mem_used, mem_total = [x.strip() for x in line.split(",")]
                row.update({
                    "gpu_util_pct": float(util),
                    "gpu_mem_used_mb": float(mem_used),
                    "gpu_mem_total_mb": float(mem_total),
                })
            except Exception:
                # no nvidia-smi or not NVIDIA GPU; ignore
                pass
        return row

    def stop(self):
        self.stop_event.set()

    def summarize(self) -> Dict[str, Any]:
        if not self.rows:
            return {}
        import numpy as np
        def agg(key): 
            arr = [r[key] for r in self.rows if key in r]
            return (float(np.mean(arr)), float(np.max(arr))) if arr else (float("nan"), float("nan"))
        cpu_total_avg, cpu_total_max = agg("cpu_total_pct")
        cpu_proc_avg,  cpu_proc_max  = agg("cpu_proc_pct")
        rss_avg, rss_max = agg("mem_proc_rss_mb")
        sys_used_avg, sys_used_max = agg("mem_sys_used_mb")
        out = {
            "device": self.device_label,
            "cpu_total_avg_pct": cpu_total_avg, "cpu_total_max_pct": cpu_total_max,
            "cpu_proc_avg_pct":  cpu_proc_avg,  "cpu_proc_max_pct":  cpu_proc_max,
            "mem_proc_rss_avg_mb": rss_avg, "mem_proc_rss_max_mb": rss_max,
            "mem_sys_used_avg_mb": sys_used_avg, "mem_sys_used_max_mb": sys_used_max,
        }
        if any("gpu_util_pct" in r for r in self.rows):
            import numpy as np
            gutil = [r["gpu_util_pct"] for r in self.rows if "gpu_util_pct" in r]
            gmemu = [r["gpu_mem_used_mb"] for r in self.rows if "gpu_mem_used_mb" in r]
            gmemt = [r["gpu_mem_total_mb"] for r in self.rows if "gpu_mem_total_mb" in r]
            out.update({
                "gpu_util_avg_pct": float(np.mean(gutil)),
                "gpu_util_max_pct": float(np.max(gutil)),
                "gpu_mem_used_avg_mb": float(np.mean(gmemu)),
                "gpu_mem_used_max_mb": float(np.max(gmemu)),
                "gpu_mem_total_mb": float(np.max(gmemt)) if gmemt else float("nan"),
            })
        return out


def load_streams_from_file(path: str) -> list[str]:
    """
    Read stream sources from a text file.
    - One source per line
    - Supports comments (#, //)
    - Expands ~ and $VARS
    - Expands glob patterns for local files (*.mp4, etc.)
    - Deduplicates while preserving order
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"streams file not found: {p}")
    out, seen = [], set()
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        s = os.path.expandvars(os.path.expanduser(s))
        if "://" in s:  # RTSP/HTTP — keep as-is
            cand = [s]
        else:
            # Expand globs for local files
            cand = sorted(glob.glob(s)) if any(ch in s for ch in "*?[]") else [s]
        for c in cand:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
    return out


def plot_stream_fps(prefix: str, device: str, summaries: List[StreamMetrics], out_dir: str = "multistrean_benchmark_20s"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    ids = [m.stream_id for m in summaries]
    fps = [m.summary()["fps_processed"] for m in summaries]
    plt.figure()
    plt.bar(ids, fps)
    plt.xlabel("Stream ID")
    plt.ylabel("FPS (processed)")
    plt.title(f"{device.upper()} per-stream FPS")
    path = os.path.join(out_dir, f"{prefix}_{device}_stream_fps.png")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path

def plot_stream_latency_mean(prefix: str, device: str, summaries: List[StreamMetrics], out_dir: str = "multistrean_benchmark_20s"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    ids = [m.stream_id for m in summaries]
    means = [m.summary()["lat_total_mean_ms"] for m in summaries]
    plt.figure()
    plt.bar(ids, means)
    plt.xlabel("Stream ID")
    plt.ylabel("Mean latency (ms)")
    plt.title(f"{device.upper()} per-stream mean latency")
    path = os.path.join(out_dir, f"{prefix}_{device}_stream_latency_mean.png")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path

def plot_device_latency_percentiles(prefix: str, device: str, dev_sum: DeviceSummary, out_dir: str = "multistrean_benchmark_20s"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    labels = ["mean", "p50", "p90", "p99"]
    vals = [
        dev_sum.lat_total_mean_ms,
        dev_sum.lat_total_p50_ms,
        dev_sum.lat_total_p90_ms,
        dev_sum.lat_total_p99_ms,
    ]
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Latency (ms)")
    plt.title(f"{device.upper()} latency summary")
    path = os.path.join(out_dir, f"{prefix}_{device}_latency_summary.png")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path

def plot_resources_timeseries(prefix: str, device: str, rows: List[Dict[str, Any]], out_dir: str = "multistrean_benchmark_20s"):
    if not rows:
        return []
    import os
    os.makedirs(out_dir, exist_ok=True)
    # Time base in seconds
    t0 = rows[0]["ts"]
    ts = [r["ts"] - t0 for r in rows]

    saved = []

    # System CPU %
    plt.figure()
    plt.plot(ts, [r.get("cpu_total_pct", float("nan")) for r in rows])
    plt.xlabel("Time (s)")
    plt.ylabel("CPU total (%)")
    plt.title(f"{device.upper()} CPU total %")
    p = os.path.join(out_dir, f"{prefix}_{device}_res_cpu_total.png")
    plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    # Process CPU %
    plt.figure()
    plt.plot(ts, [r.get("cpu_proc_pct", float("nan")) for r in rows])
    plt.xlabel("Time (s)")
    plt.ylabel("CPU process (%)")
    plt.title(f"{device.upper()} CPU process %")
    p = os.path.join(out_dir, f"{prefix}_{device}_res_cpu_proc.png")
    plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    # Process RSS MB
    plt.figure()
    plt.plot(ts, [r.get("mem_proc_rss_mb", float("nan")) for r in rows])
    plt.xlabel("Time (s)")
    plt.ylabel("RSS (MB)")
    plt.title(f"{device.upper()} Process RSS")
    p = os.path.join(out_dir, f"{prefix}_{device}_res_rss.png")
    plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    # GPU util % (if present)
    if any("gpu_util_pct" in r for r in rows):
        plt.figure()
        plt.plot(ts, [r.get("gpu_util_pct", float("nan")) for r in rows])
        plt.xlabel("Time (s)")
        plt.ylabel("GPU util (%)")
        plt.title(f"{device.upper()} GPU util %")
        p = os.path.join(out_dir, f"{prefix}_{device}_res_gpu_util.png")
        plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    # GPU mem used MB (if present)
    if any("gpu_mem_used_mb" in r for r in rows):
        plt.figure()
        plt.plot(ts, [r.get("gpu_mem_used_mb", float("nan")) for r in rows])
        plt.xlabel("Time (s)")
        plt.ylabel("GPU mem used (MB)")
        plt.title(f"{device.upper()} GPU memory used")
        p = os.path.join(out_dir, f"{prefix}_{device}_res_gpu_mem.png")
        plt.tight_layout(); plt.savefig(p); plt.close(); saved.append(p)

    return saved



# ---------- Data holders ----------
@dataclass
class StreamMetrics:
    stream_id: int
    source: str
    processed: int = 0
    dropped: int = 0
    decode_errors: int = 0
    warmup_frames: int = 0
    # latencies (seconds)
    t_decode: List[float] = field(default_factory=list)
    t_infer_pre: List[float] = field(default_factory=list)
    t_infer: List[float] = field(default_factory=list)
    t_infer_post: List[float] = field(default_factory=list)
    t_total: List[float] = field(default_factory=list)  # decode + infer total
    
    def add(self, td: float, tpre: float, tinf: float, tpost: float):
        self.t_decode.append(td)
        self.t_infer_pre.append(tpre)
        self.t_infer.append(tinf)
        self.t_infer_post.append(tpost)
        self.t_total.append(td + tpre + tinf + tpost)
        self.processed += 1

    def summary(self) -> Dict[str, Any]:
        wall = sum(self.t_total)  # processed wall time for counted frames
        pcts_total = percentiles(self.t_total)
        pcts_infer = percentiles(self.t_infer)
        return {
            "stream_id": self.stream_id,
            "source": self.source,
            "frames": self.processed,
            "dropped": self.dropped,
            "decode_errors": self.decode_errors,
            "fps_processed": (self.processed / wall) if wall > 0 else 0.0,
            "lat_total_mean_ms": (np.mean(self.t_total) * 1000.0) if self.t_total else float("nan"),
            **{f"lat_total_{k}_ms": v * 1000.0 for k, v in pcts_total.items()},
            "lat_infer_mean_ms": (np.mean(self.t_infer) * 1000.0) if self.t_infer else float("nan"),
            **{f"lat_infer_{k}_ms": v * 1000.0 for k, v in pcts_infer.items()},
        }


@dataclass
class DeviceSummary:
    device_label: str
    duration_s: float
    total_frames: int
    total_dropped: int
    total_decode_errors: int
    overall_fps: float
    lat_total_mean_ms: float
    lat_total_p50_ms: float
    lat_total_p90_ms: float
    lat_total_p99_ms: float


# ---------- Video capture thread per stream ----------
class CaptureThread(threading.Thread):
    def __init__(
        self,
        stream_id: int,
        source: str,
        out_queue: "queue.Queue[Tuple[int, np.ndarray, float]]",
        stop_event: threading.Event,
        reconnect_secs: float = 3.0,
        drop_policy: str = "latest",  # latest|block
    ):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.source = source
        self.q = out_queue
        self.stop_event = stop_event
        self.reconnect_secs = reconnect_secs
        self.drop_policy = drop_policy
        self.metrics = StreamMetrics(stream_id=stream_id, source=source)

    def run(self):
        cap = None
        last_open_try = 0.0
        while not self.stop_event.is_set():
            # Ensure VideoCapture is open
            if cap is None or not cap.isOpened():
                now_t = time.time()
                if now_t - last_open_try < self.reconnect_secs:
                    time.sleep(0.05)
                    continue
                last_open_try = now_t
                cap = cv2.VideoCapture(self.source)
                # For RTSP low latency, reduce internal buffering
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            t0 = now()
            ok, frame = cap.read()
            t1 = now()
            if not ok or frame is None:
                self.metrics.decode_errors += 1
                # Reopen on failure
                cap.release()
                cap = None
                time.sleep(0.1)
                continue

            # Push frame with decode time
            item = (self.stream_id, frame, t1 - t0)
            try:
                if self.drop_policy == "latest":
                    # Non-blocking put; drop the frame if queue full
                    self.q.put_nowait(item)
                else:
                    self.q.put(item, timeout=0.1)
            except queue.Full:
                # Drop most recent to keep latency low
                self.metrics.dropped += 1
                # Overwrite existing by clearing and pushing once
                try:
                    _ = self.q.get_nowait()
                except Exception:
                    pass
                try:
                    self.q.put_nowait(item)
                except Exception:
                    self.metrics.dropped += 1
                    pass

        # Cleanup
        if cap is not None:
            cap.release()


# ---------- Engine wrappers ----------
# ---------- Engine wrappers ----------
import inspect

class EngineWrapper:
    """Adapter to unify GPU/NPU engine .detect() outputs and pass optional kwargs if supported."""
    def __init__(self, kind: str, engine: Any, img_size: Tuple[int, int] | None = None,
                 conf: float | None = None, iou: float | None = None):
        self.kind = kind
        self.engine = engine
        self.img_size = img_size
        self.conf = conf
        self.iou = iou

        # Introspect detect() signature once
        self._detect_sig = None
        if hasattr(self.engine, "detect"):
            try:
                self._detect_sig = inspect.signature(self.engine.detect)
            except Exception:
                self._detect_sig = None

    def _build_call_kwargs(self) -> Dict[str, Any]:
        """Match common kw names if detect() supports them."""
        if self._detect_sig is None:
            return {}
        params = self._detect_sig.parameters
        kw = {}
        # confidence
        if self.conf is not None:
            for name in ("conf", "conf_thres", "conf_thresh", "confidence"):
                if name in params:
                    kw[name] = self.conf
                    break
        # iou
        if self.iou is not None:
            for name in ("iou", "iou_thres", "iou_thresh"):
                if name in params:
                    kw[name] = self.iou
                    break
        # img size
        if self.img_size is not None:
            for name in ("img_size", "size", "input_size"):
                if name in params:
                    kw[name] = self.img_size
                    break
        return kw

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Returns: (detections, timings_dict) where timings has keys pre/infer/post (seconds).
        If the underlying engine returns only detections, we synthesize timings around the call.
        """
        call_kwargs = self._build_call_kwargs()
        t0 = now()
        try:
            out = self.engine.detect(frame, **call_kwargs)
        except TypeError:
            # If we guessed wrong about kwargs, fall back to positional-only call
            out = self.engine.detect(frame)
        t1 = now()

        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            dets, tdict = out
            pre = float(tdict.get("pre", 0.0))
            post = float(tdict.get("post", 0.0))
            infer = float(tdict.get("infer", max(0.0, (t1 - t0) - pre - post)))
            return dets, {"pre": pre, "infer": infer, "post": post}
        else:
            return out, {"pre": 0.0, "infer": (t1 - t0), "post": 0.0}



def make_npu_engine(mxq_path: str, img_size: Tuple[int, int], conf: float, iou: float, class_names: List[str]):
    if NPUInferenceEngine is None or InferenceConfig is None:
        raise RuntimeError("npu_inference module not available. Ensure npu_inference.py is importable.")

    cfg = InferenceConfig(mxq_path)  # no kwargs; set attributes directly
    cfg.model_path = mxq_path
    cfg.img_size = img_size
    cfg.num_classes = len(class_names)
    cfg.num_layers = 3
    cfg.reg_max = 16
    cfg.conf_threshold = conf
    cfg.iou_threshold = iou
    cfg.use_global8_core = True  # or False if you want per-core mode
    cfg.device_id = 0

    engine = NPUInferenceEngine(cfg, class_names=class_names)
    if hasattr(engine, "initialize"):
        try:
            engine.initialize()
        except Exception as e:
            raise RuntimeError(f"NPU engine initialize() failed: {e}")
    return EngineWrapper("npu", engine, img_size=img_size, conf=conf, iou=iou)




def make_gpu_engine(model_path: str, img_size: Tuple[int, int], conf: float, iou: float, class_names: List[str]):
    if GPUYOLOv8Engine is None:
        raise RuntimeError("gpu_yolov8_inference module not available. Ensure gpu_yolov8_inference.py is importable.")
    # Construct with the safest minimal signature (no unsupported kwargs)
    try:
        engine = GPUYOLOv8Engine(model_path)
    except TypeError:
        # Some implementations require keyword
        engine = GPUYOLOv8Engine(model_path=model_path)

    # Try to push configuration into the engine if it exposes setters/fields
    for name, val in (("class_names", class_names),):
        if hasattr(engine, name):
            try: setattr(engine, name, val)
            except Exception: pass

    for setter, val in (("set_conf_thresh", conf), ("set_conf_thres", conf), ("set_confidence", conf)):
        if hasattr(engine, setter):
            try: getattr(engine, setter)(val)
            except Exception: pass
    for setter, val in (("set_iou_thresh", iou), ("set_iou_thres", iou)):
        if hasattr(engine, setter):
            try: getattr(engine, setter)(val)
            except Exception: pass
    for setter, val in (("set_input_size", img_size), ("set_img_size", img_size), ("set_size", img_size)):
        if hasattr(engine, setter):
            try: getattr(engine, setter)(val)
            except Exception: pass

    # Initialize if engine provides initialize()
    if hasattr(engine, "initialize"):
        try: engine.initialize()
        except Exception: pass

    return EngineWrapper("gpu", engine, img_size=img_size, conf=conf, iou=iou)




# ---------- Worker that binds a stream to an engine ----------
class InferenceWorker(threading.Thread):
    def __init__(
        self,
        stream_id: int,
        in_queue: "queue.Queue[Tuple[int, np.ndarray, float]]",
        engine: EngineWrapper,
        stop_event: threading.Event,
        warmup_frames: int = 10,
        metrics: Optional[StreamMetrics] = None,
    ):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.in_queue = in_queue
        self.engine = engine
        self.stop_event = stop_event
        self.warmup_frames = warmup_frames
        self.metrics = metrics or StreamMetrics(stream_id=stream_id, source=f"stream-{stream_id}")

    def run(self):
        warm = self.warmup_frames
        while not self.stop_event.is_set():
            try:
                sid, frame, t_decode = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                continue

            if warm > 0:
                _ = self.engine.detect(frame)  # not recorded
                warm -= 1
                self.metrics.warmup_frames += 1
                continue

            t0 = now()
            detections, timings = self.engine.detect(frame)
            t1 = now()
            pre = float(timings.get("pre", 0.0))
            inf = float(timings.get("infer", (t1 - t0)))
            post = float(timings.get("post", 0.0))
            self.metrics.add(td=t_decode, tpre=pre, tinf=inf, tpost=post)


# ---------- Orchestrator ----------
def run_device_multistream(
    device_label: str,
    engine_factory,
    model_path: str,
    streams: List[str],
    duration_s: int,
    img_size: Tuple[int, int],
    conf: float,
    iou: float,
    class_names: List[str],
    engine_instances: int = 1,
    warmup_frames: int = 10,
    drop_policy: str = "latest",
    save_prefix: str = "run",        # <--- NEW
) -> Tuple[List[StreamMetrics], DeviceSummary]:
    """
    Runs multistream load on a given device.
    engine_factory: callable -> EngineWrapper
    """
    n_streams = len(streams)
    stop_event = threading.Event()

    # Resource monitor
    mon = ResourceMonitor(device_label, interval=0.5, with_gpu=(device_label=="gpu"))
    mon.start()

    # 1) Create engine pool first (fail fast, no orphan capture threads)
    engines = [engine_factory(model_path, img_size, conf, iou, class_names) for _ in range(engine_instances)]

    # 2) Create one queue per stream
    queues = [queue.Queue(maxsize=1) for _ in range(n_streams)]

    # 3) Start capture threads
    caps = []
    stream_metrics = []
    for i, src in enumerate(streams):
        ct = CaptureThread(i, src, queues[i], stop_event, drop_policy=drop_policy)
        caps.append(ct)
        stream_metrics.append(ct.metrics)
        ct.start()

    # 4) Start inference workers
    workers = []
    for i in range(n_streams):
        eng = engines[i % engine_instances]
        wk = InferenceWorker(
            stream_id=i,
            in_queue=queues[i],
            engine=eng,
            stop_event=stop_event,
            warmup_frames=warmup_frames,
            metrics=stream_metrics[i],
        )
        workers.append(wk)
        wk.start()

    # Run for duration
    t_start = now()
    try:
        while (now() - t_start) < duration_s:
            time.sleep(0.25)
    finally:
        stop_event.set()
        # Join threads
        for ct in caps:
            ct.join(timeout=2.0)
        for wk in workers:
            wk.join(timeout=2.0)

        mon.stop()

    # Device-level summary
    all_total_lat = []
    total_frames = 0
    total_dropped = 0
    total_decode_errors = 0
    duration = now() - t_start
    for m in stream_metrics:
        all_total_lat.extend(m.t_total)
        total_frames += m.processed
        total_dropped += m.dropped
        total_decode_errors += m.decode_errors

    overall_fps = total_frames / duration if duration > 0 else 0.0
    pcts = percentiles(all_total_lat)
    dev_sum = DeviceSummary(
        device_label=device_label,
        duration_s=duration,
        total_frames=total_frames,
        total_dropped=total_dropped,
        total_decode_errors=total_decode_errors,
        overall_fps=overall_fps,
        lat_total_mean_ms=(np.mean(all_total_lat) * 1000.0) if all_total_lat else float("nan"),
        lat_total_p50_ms=pcts.get("p50", float("nan")) * 1000.0,
        lat_total_p90_ms=pcts.get("p90", float("nan")) * 1000.0,
        lat_total_p99_ms=pcts.get("p99", float("nan")) * 1000.0,
    )

    # Summarize & persist resources
    res_summary = mon.summarize()
    if res_summary:
        print(f"\n[{device_label.upper()}] Resource summary:")
        print(f"  CPU total avg/max: {res_summary['cpu_total_avg_pct']:.1f}% / {res_summary['cpu_total_max_pct']:.1f}%")
        print(f"  CPU proc  avg/max: {res_summary['cpu_proc_avg_pct']:.1f}% / {res_summary['cpu_proc_max_pct']:.1f}%")
        print(f"  RSS (proc) avg/max: {res_summary['mem_proc_rss_avg_mb']:.1f} / {res_summary['mem_proc_rss_max_mb']:.1f} MB")
        print(f"  Sys RAM  avg/max: {res_summary['mem_sys_used_avg_mb']:.0f} / {res_summary['mem_sys_used_max_mb']:.0f} MB")
        if 'gpu_util_avg_pct' in res_summary:
            print(f"  GPU util  avg/max: {res_summary['gpu_util_avg_pct']:.1f}% / {res_summary['gpu_util_max_pct']:.1f}%")
            print(f"  GPU mem used avg/max: {res_summary['gpu_mem_used_avg_mb']:.0f} / {res_summary['gpu_mem_used_max_mb']:.0f} MB")

    # Save CSV of resource samples
    save_resources_csv("run", device_label, mon.rows)
    # Make plots
    try:
        plot_stream_fps(save_prefix, device_label, stream_metrics)
        plot_stream_latency_mean(save_prefix, device_label, stream_metrics)
        plot_device_latency_percentiles(save_prefix, device_label, dev_sum)
        plot_resources_timeseries(save_prefix, device_label, mon.rows)
    except Exception as e:
        print(f"[WARN] plotting failed: {e}")

    # return stream_metrics, dev_sum
    return stream_metrics, dev_sum, mon.rows


def print_device_report(device: str, summaries: List[StreamMetrics], dev_sum: DeviceSummary):
    print(f"\n===== {device.upper()} MULTISTREAM SUMMARY =====")
    print(f"Duration: {dev_sum.duration_s:.1f}s | Total frames: {dev_sum.total_frames} "
          f"| Dropped: {dev_sum.total_dropped} | Decode errors: {dev_sum.total_decode_errors}")
    print(f"Overall FPS: {dev_sum.overall_fps:.2f}")
    print(f"Latency total mean/p50/p90/p99 (ms): "
          f"{dev_sum.lat_total_mean_ms:.1f} / {dev_sum.lat_total_p50_ms:.1f} / "
          f"{dev_sum.lat_total_p90_ms:.1f} / {dev_sum.lat_total_p99_ms:.1f}")
    print("\nPer-stream:")
    print(f"{'ID':>2}  {'FPS':>6}  {'Frames':>7}  {'Drop':>5}  {'DecErr':>6}  {'Mean(ms)':>9}  {'p50':>7}  {'p90':>7}  {'p99':>7}  Source")
    for m in summaries:
        s = m.summary()
        print(f"{s['stream_id']:>2}  {s['fps_processed']:>6.2f}  {s['frames']:>7}  {s['dropped']:>5}  {s['decode_errors']:>6}  "
              f"{s['lat_total_mean_ms']:>9.1f}  {s['lat_total_p50_ms']:>7.1f}  {s['lat_total_p90_ms']:>7.1f}  {s['lat_total_p99_ms']:>7.1f}  {s['source']}")



def save_resources_csv(prefix: str, device: str, rows: List[Dict[str, Any]], out_dir: str = "multistrean_benchmark_20s"):
    import csv, os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_{device}_resources.csv")
    if not rows:
        return path
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path

def save_csv(prefix: str, device: str, summaries: List[StreamMetrics], dev_sum: DeviceSummary, out_dir: str = "multistrean_benchmark_20s"):
    import csv, os
    os.makedirs(out_dir, exist_ok=True)
    # Per-stream
    streams_csv = os.path.join(out_dir, f"{prefix}_{device}_streams.csv")
    with open(streams_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(StreamMetrics(0, "").summary().keys()))
        writer.writeheader()
        for m in summaries:
            writer.writerow(m.summary())
    # Device summary
    dev_csv = os.path.join(out_dir, f"{prefix}_{device}_device.csv")
    with open(dev_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(dev_sum).keys()))
        writer.writeheader()
        writer.writerow(asdict(dev_sum))
    return streams_csv, dev_csv


# ---------- CLI ----------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser("YOLOv8 GPU vs NPU Multistream Benchmark")
    ap.add_argument("--streams", nargs="+", default=DEFAULT_STREAMS,
                    help="Space-separated list of RTSP/file paths. Tip: repeat a short file to simulate more streams.")
    ap.add_argument("--streams-file", type=str, default=None,
                    help="Optional text file with one stream URL per line.")
    ap.add_argument("--device", choices=["gpu", "npu", "both-seq"], default="both-seq",
                    help="Which device(s) to benchmark. 'both-seq' runs GPU then NPU, avoiding contention.")
    ap.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds per device.")
    ap.add_argument("--img-size", type=int, nargs=2, default=[640, 640], help="Model input size WxH.")
    ap.add_argument("--conf", type=float, default=0.55, help="Confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.55, help="IoU threshold.")
    ap.add_argument("--npu-model", type=str, default=DEFAULT_NPU_MXQ, help=".mxq path for NPU.")
    ap.add_argument("--gpu-model", type=str, default=DEFAULT_GPU_TORCHSCRIPT, help="TorchScript .pt/.ts path for GPU.")
    ap.add_argument("--npu-instances", type=int, default=1, help="Number of parallel NPU engine instances.")
    ap.add_argument("--gpu-instances", type=int, default=1, help="Number of parallel GPU engine instances.")
    ap.add_argument("--warmup-frames", type=int, default=10, help="Warmup frames per stream (excluded from metrics).")
    ap.add_argument("--drop-policy", choices=["latest", "block"], default="latest",
                    help="On queue full: 'latest' drops newest to keep latency low; 'block' waits.")
    ap.add_argument("--save-csv-prefix", type=str, default="run",
                    help="CSV file prefix saved into ./multistrean_benchmark_20s_v2/")
    return ap.parse_args()


def main():
    args = parse_args()

    # Build streams list
    streams = list(args.streams)
    if args.streams_file:
        try:
            streams = load_streams_from_file(args.streams_file)
        except Exception as e:
            print(f"[ERR] failed to read --streams-file: {e}")
            sys.exit(2)

    if not streams and args.streams:
        streams = list(args.streams)

    if not streams:
        print("[ERR] No streams provided. Use --streams or --streams-file streams.txt")
        sys.exit(2)

    print(f"[INFO] Loaded {len(streams)} sources. First 3: {streams[:3]}")

    img_size = (int(args.img_size[0]), int(args.img_size[1]))

    # Run GPU
    if args.device in ("gpu", "both-seq"):
        print("\n=== Running GPU multistream ===")
        g_summaries, g_dev, g_res = run_device_multistream(
            device_label="gpu",
            engine_factory=lambda path, sz, cf, iou, names: make_gpu_engine(path, sz, cf, iou, names),
            model_path=args.gpu_model,
            streams=streams,
            duration_s=args.duration,
            img_size=img_size,
            conf=args.conf,
            iou=args.iou,
            class_names=DEFAULT_CLASS_NAMES,
            engine_instances=args.gpu_instances,
            warmup_frames=args.warmup_frames,
            drop_policy=args.drop_policy,
            save_prefix=args.save_csv_prefix,  
        )
        print_device_report("gpu", g_summaries, g_dev)
        save_csv(args.save_csv_prefix, "gpu", g_summaries, g_dev)

    # Run NPU
    if args.device in ("npu", "both-seq"):
        print("\n=== Running NPU multistream ===")
        n_summaries, n_dev, n_res = run_device_multistream(
            device_label="npu",
            engine_factory=lambda path, sz, cf, iou, names: make_npu_engine(path, sz, cf, iou, names),
            model_path=args.npu_model,
            streams=streams,
            duration_s=args.duration,
            img_size=img_size,
            conf=args.conf,
            iou=args.iou,
            class_names=DEFAULT_CLASS_NAMES,
            engine_instances=args.npu_instances,
            warmup_frames=args.warmup_frames,
            drop_policy=args.drop_policy,
            save_prefix=args.save_csv_prefix,
        )
        print_device_report("npu", n_summaries, n_dev)
        save_csv(args.save_csv_prefix, "npu", n_summaries, n_dev)
        # Combined plots
        plot_compare_latency(args.save_csv_prefix, g_dev, n_dev)
        plot_compare_overall_fps(args.save_csv_prefix, g_dev, n_dev)
        plot_compare_stream_fps(args.save_csv_prefix, g_summaries, n_summaries)
        plot_compare_resources(args.save_csv_prefix, g_res, n_res)

if __name__ == "__main__":
    main()
