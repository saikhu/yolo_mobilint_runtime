"""
NPU Inference Module for YOLOv11 Object Detection
Production-ready module for Mobilint NPU inference.

This implementation is compatible with YOLOv11 MXQ models that output
separate class and box(DFL) heads, often in HWC/NHWC layouts.

Public API matches the YOLOv8 module:
- InferenceConfig
- Detection
- NPUInferenceEngine

Key fix included:
- Enforces stable scale ordering to match anchor generation order
  (prevents bbox drifting/misalignment).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
import maccel

# Optional imports for postprocessing
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Postprocessing will be limited.")

logger = logging.getLogger(__name__)


# -------------------------
# Data structures
# -------------------------

@dataclass
class InferenceConfig:
    """Configuration for NPU inference."""
    model_path: str
    img_size: Tuple[int, int] = (1280, 1280)      # Your YOLOv11 MXQ test uses 1280
    num_classes: Optional[int] = None            # Can be inferred from outputs
    num_layers: int = 3                          # Usually 3 scales
    reg_max: int = 16                            # DFL bins
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    use_global8_core: bool = True                # Your MXQ works in GLOBAL8
    device_id: int = 0


@dataclass
class Detection:
    """Single detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: Optional[str] = None


# -------------------------
# Engine
# -------------------------

class NPUInferenceEngine:
    """
    NPU Inference Engine for YOLOv11 object detection.

    Handles:
    - Model loading (MAccel)
    - Preprocess (letterbox + normalize)
    - Inference
    - Postprocess (anchor gen + DFL decode + NMS)
    """

    # Override from caller if your YOLOv11 model uses a different taxonomy.
    DEFAULT_CLASS_NAMES = [
        "cement_truck", "compactor", "dump_truck", "excavator", "grader",
        "mobile_crane", "tower_crane", "Crane_Hook", "worker", "Hardhat",
        "Red_Hardhat", "scaffolds", "Lifted Load", "Hook"
    ]

    def __init__(self, config: InferenceConfig, class_names: Optional[List[str]] = None):
        self.config = config
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES

        self._model: Optional[maccel.Model] = None
        self._accelerator: Optional[maccel.Accelerator] = None
        self._initialized: bool = False

        logger.info(f"Initializing YOLOv11 NPU Inference Engine with model: {config.model_path}")

    # -------------------------
    # Lifecycle
    # -------------------------

    def initialize(self) -> None:
        """Initialize NPU hardware and load model."""
        try:
            self._accelerator = maccel.Accelerator(self.config.device_id)
            logger.info(f"NPU Accelerator initialized on device {self.config.device_id}")

            model_config = maccel.ModelConfig()
            if self.config.use_global8_core:
                model_config.set_global8_core_mode()
                logger.info("Using Global8 core mode")

            self._model = maccel.Model(self.config.model_path, model_config)
            self._model.launch(self._accelerator)

            self._initialized = True
            logger.info("YOLOv11 model loaded and launched successfully")

        except Exception as e:
            logger.error(f"Failed to initialize NPU: {str(e)}")
            raise RuntimeError(f"NPU initialization failed: {str(e)}")

    def cleanup(self) -> None:
        """Clean up NPU resources."""
        if self._model:
            try:
                self._model.dispose()
                logger.info("Model disposed successfully")
            except Exception as e:
                logger.error(f"Error disposing model: {str(e)}")

        self._model = None
        self._accelerator = None
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    # -------------------------
    # Pre / Infer / Post
    # -------------------------

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess image for YOLOv11 inference (letterbox + normalize).
        Returns NCHW float32 in [0,1].
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if image.ndim != 3:
            raise ValueError(f"Expected 3D array (H,W,C), got shape {image.shape}")

        h0, w0 = image.shape[:2]
        target_h, target_w = self.config.img_size

        # scale ratio
        r = min(target_h / h0, target_w / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))

        # padding
        dw = target_w - new_unpad[0]
        dh = target_h - new_unpad[1]
        dw /= 2
        dh /= 2

        if (image.shape[1], image.shape[0]) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # normalize + HWC->CHW + add batch
        img = image.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]

        metadata = {
            "original_shape": (h0, w0),
            "scale": r,
            "padding": {"top": top, "left": left},
        }
        return img, metadata

    def infer(self, preprocessed_image: np.ndarray) -> List[np.ndarray]:
        """Run inference on NPU."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        try:
            return self._model.infer(preprocessed_image)
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"NPU inference failed: {str(e)}")

    def postprocess(self, raw_outputs: List[np.ndarray], metadata: Dict[str, Any]) -> List[Detection]:
        """Convert raw outputs to Detection objects in original image space."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for postprocessing. Install with: pip install torch")

        dets = self._decode_outputs(raw_outputs)
        if len(dets) == 0:
            return []

        scale = metadata["scale"]
        padding = metadata["padding"]
        h0, w0 = metadata["original_shape"]

        results: List[Detection] = []
        for x1, y1, x2, y2, conf, cls_id in dets:
            # Undo letterbox
            x1 = (x1 - padding["left"]) / scale
            y1 = (y1 - padding["top"]) / scale
            x2 = (x2 - padding["left"]) / scale
            y2 = (y2 - padding["top"]) / scale

            # Clip
            x1 = max(0, min(x1, w0))
            y1 = max(0, min(y1, h0))
            x2 = max(0, min(x2, w0))
            y2 = max(0, min(y2, h0))

            cid = int(cls_id)
            cname = self.class_names[cid] if cid < len(self.class_names) else None

            results.append(
                Detection(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=float(conf),
                    class_id=cid,
                    class_name=cname
                )
            )

        return results

    # -------------------------
    # Decode utilities
    # -------------------------

    def _to_nchw(self, arr: np.ndarray) -> "torch.Tensor":
        """
        Convert various output layouts to torch NCHW.

        Supports:
          - HWC
          - CHW
          - NHWC
          - NCHW

        Your MXQ test shows outputs like:
          (40,40,14), (40,40,64), ...
        which are HWC.
        """
        t = torch.from_numpy(arr).to(torch.float32)

        reg_c = self.config.reg_max * 4
        cls_c = self.config.num_classes

        def looks_like_channel_dim(x: int) -> bool:
            return x == reg_c or (cls_c is not None and x == cls_c)

        if t.ndim == 3:
            a, b, c = t.shape
            # Likely HWC
            if looks_like_channel_dim(c):
                t = t.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
            else:
                # Assume CHW
                t = t.unsqueeze(0)
        elif t.ndim == 4:
            n, a, b, c = t.shape
            # If last dim looks like channels -> NHWC
            if looks_like_channel_dim(c):
                t = t.permute(0, 3, 1, 2)
            # else assume NCHW already
        else:
            raise ValueError(f"Unexpected output ndim: {t.ndim}, shape={arr.shape}")

        return t

    def _decode_outputs(self, raw_outputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Decode YOLOv11 outputs with anchors, DFL decoding, and NMS.

        Returns array of [x1, y1, x2, y2, conf, cls_id].

        IMPORTANT:
        Enforces stable scale ordering to match _make_anchors():
          strides [8,16,32] -> grids [H/8, H/16, H/32]
        This prevents anchor/prediction order mismatch and bbox misplacement.
        """
        device = torch.device("cpu")

        reg_c = self.config.reg_max * 4

        # Convert outputs to NCHW
        tensors = [self._to_nchw(o).to(device) for o in raw_outputs]

        det_map: Dict[Tuple[int, int], "torch.Tensor"] = {}
        cls_map: Dict[Tuple[int, int], "torch.Tensor"] = {}

        inferred_num_classes: Optional[int] = self.config.num_classes

        for t in tensors:
            if t.ndim != 4:
                continue
            _, c, h, w = t.shape
            key = (h, w)

            if c == reg_c:
                det_map[key] = t
            else:
                cls_map[key] = t
                if inferred_num_classes is None:
                    inferred_num_classes = int(c)

        if inferred_num_classes is None:
            inferred_num_classes = 1

        # Keys present in both heads
        common_keys = [k for k in det_map.keys() if k in cls_map]
        if not common_keys:
            logger.warning("No matching det/cls scales found in outputs.")
            return []

        # Sort keys to match anchor order (stride ascending)
        # stride = img_h / grid_h
        img_h = self.config.img_size[0]
        common_keys = sorted(common_keys, key=lambda k: (img_h // k[0]))

        paired = []
        for key in common_keys:
            det = det_map[key]
            cls = cls_map[key]

            if self.config.num_classes is not None and cls.shape[1] != self.config.num_classes:
                logger.warning(
                    f"Class head channels ({cls.shape[1]}) != config.num_classes ({self.config.num_classes}). "
                    "Proceeding with head channels."
                )

            # (1, reg_c + num_classes, H*W)
            paired.append(torch.cat((det, cls), dim=1).flatten(2))

        if not paired:
            return []

        # Concatenate all scales along anchor dimension
        # Result: (1, reg_c + num_classes, A)
        out = torch.cat(paired, dim=2)
        batch_out = out[0]  # (C, A)

        # Split
        box_raw = batch_out[:reg_c]              # (reg_c, A)
        cls_raw = batch_out[reg_c:]              # (num_classes, A)

        # Class probabilities
        cls_prob = cls_raw.sigmoid()
        conf, cls_idx = torch.max(cls_prob, dim=0)  # (A), (A)

        # Build anchors/strides in SAME order as common_keys sorting expectation
        anchors, strides = self._make_anchors(self.config.num_layers, self.config.img_size)
        anchors = anchors.to(device)
        strides = strides.to(device)

        A_pred = box_raw.shape[1]
        A_anchor = anchors.shape[1]

        if A_pred != A_anchor:
            logger.warning(f"Anchor count mismatch: A_pred={A_pred}, A_anchor={A_anchor}. Aligning safely.")

            if A_pred > A_anchor:
                # Truncate predictions
                box_raw = box_raw[:, :A_anchor]
                conf = conf[:A_anchor]
                cls_idx = cls_idx[:A_anchor]
                A_pred = A_anchor
            else:
                # Truncate anchors
                anchors = anchors[:, :A_pred]
                strides = strides[:, :A_pred]
                A_anchor = A_pred

        # Apply threshold
        keep = conf > self.config.conf_threshold
        if keep.sum() == 0:
            return []

        # Filter
        box_raw = box_raw[:, keep]
        conf_k = conf[keep]
        cls_k = cls_idx[keep]

        anchors_k = anchors[:, keep]
        strides_k = strides[:, keep]

        # Decode DFL distances -> boxes on feature grid
        dist = self._decode_dfl(box_raw.unsqueeze(0), self.config.reg_max).squeeze(0)  # (4, K)

        # dist2bbox expects (K,4) and (K,2)
        boxes = self._dist2bbox(dist.T, anchors_k.T, xywh=False) * strides_k.T

        dets = torch.cat(
            [boxes, conf_k.unsqueeze(1), cls_k.float().unsqueeze(1)],
            dim=1
        ).cpu().numpy()

        return self._nms_numpy(dets, self.config.iou_threshold, self.config.conf_threshold)

    def _make_anchors(
        self,
        nl: int = 3,
        img_size: Tuple[int, int] = (1280, 1280),
        offset: float = 0.5
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Generate anchor points and stride tensors for anchor-free heads.

        For nl=3:
          strides = [8,16,32]
        """
        imh, imw = img_size

        anchor_points = []
        stride_tensor = []

        strides = [2 ** (3 + i) for i in range(nl)]  # 8, 16, 32

        for strd in strides:
            ny = imh // strd
            nx = imw // strd

            sy = torch.arange(ny, dtype=torch.float32) + offset
            sx = torch.arange(nx, dtype=torch.float32) + offset
            yv, xv = torch.meshgrid(sy, sx, indexing="ij")

            anchor_points.append(torch.stack((xv, yv), -1).reshape(-1, 2))
            stride_tensor.append(torch.full((ny * nx, 1), strd, dtype=torch.float32))

        anchors = torch.cat(anchor_points, dim=0).T  # (2, A)
        strides = torch.cat(stride_tensor, dim=0).T  # (1, A)

        return anchors, strides

    def _decode_dfl(self, x: "torch.Tensor", reg_max: int = 16) -> "torch.Tensor":
        """
        Decode Distribution Focal Loss (DFL) predictions.

        x: (B, 4*reg_max, A)
        returns: (B, 4, A)
        """
        b, _, a = x.shape
        dfl_weight = torch.arange(reg_max, dtype=torch.float32).reshape(1, reg_max, 1, 1)

        x = x.view(b, 4, reg_max, a).transpose(2, 1).softmax(1)
        return F.conv2d(x, dfl_weight).view(b, 4, a)

    def _dist2bbox(
        self,
        distance: "torch.Tensor",
        anchor_points: "torch.Tensor",
        xywh: bool = True,
        dim: int = -1
    ) -> "torch.Tensor":
        """
        Convert distance predictions to bounding boxes.

        distance: (K, 4)
        anchor_points: (K, 2)
        """
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb

        if xywh:
            return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
        else:
            return torch.cat((x1y1, x2y2), dim)

    def _nms_numpy(
        self,
        dets: np.ndarray,
        iou_thresh: float = 0.45,
        conf_thresh: float = 0.3
    ) -> List[np.ndarray]:
        """
        Class-agnostic NMS in NumPy.

        dets: [x1, y1, x2, y2, conf, cls]
        """
        if len(dets) == 0:
            return []

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        conf = dets[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = conf.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            if dets[i, 4] < conf_thresh:
                break

            keep.append(dets[i])

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

            if len(keep) >= 300:
                break

        return keep

    # -------------------------
    # Public pipeline
    # -------------------------

    def detect(self, image: np.ndarray) -> List[Detection]:
        preprocessed, metadata = self.preprocess(image)
        raw_outputs = self.infer(preprocessed)
        return self.postprocess(raw_outputs, metadata)

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        return [self.detect(img) for img in images]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "model_path": self.config.model_path,
            "device_id": self.config.device_id,
            "img_size": self.config.img_size,
            "num_classes": self.config.num_classes,
            "num_layers": self.config.num_layers,
            "reg_max": self.config.reg_max,
            "conf_threshold": self.config.conf_threshold,
            "iou_threshold": self.config.iou_threshold,
            "use_global8_core": self.config.use_global8_core,
        }
