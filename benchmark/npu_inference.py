"""
NPU Inference Module for YOLOv8 Object Detection
Production-ready module for Mobilint NPU inference.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import maccel

# Optional imports for postprocessing (can be made conditional)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Postprocessing will be limited.")

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for NPU inference."""
    model_path: str
    img_size: Tuple[int, int] = (640, 640)
    num_classes: int = 17
    num_layers: int = 3
    reg_max: int = 16
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    use_global8_core: bool = True
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


class NPUInferenceEngine:
    """
    NPU Inference Engine for YOLOv8 object detection.
    
    This class handles model loading, preprocessing, inference,
    and postprocessing on Mobilint NPU hardware.
    """
    
    # Default class names - can be overridden
    DEFAULT_CLASS_NAMES = [
        "backhoe_loader", "cement_truck", "compactor", "dozer", "dump_truck",
        "excavator", "grader", "mobile_crane", "tower_crane", "wheel_loader",
        "worker", "Hardhat", "Red_Hardhat", "scaffolds", "Lifted Load",
        "Crane_Hook", "Hook"
    ]
    
    def __init__(self, config: InferenceConfig, class_names: Optional[List[str]] = None):
        """
        Initialize NPU Inference Engine.
        
        Args:
            config: Inference configuration
            class_names: Optional list of class names (uses default if not provided)
        """
        self.config = config
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self._model = None
        self._accelerator = None
        self._initialized = False
        
        logger.info(f"Initializing NPU Inference Engine with model: {config.model_path}")
    
    def initialize(self) -> None:
        """
        Initialize NPU hardware and load model.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Initialize accelerator
            self._accelerator = maccel.Accelerator(self.config.device_id)
            logger.info(f"NPU Accelerator initialized on device {self.config.device_id}")
            
            # Configure model
            model_config = maccel.ModelConfig()
            if self.config.use_global8_core:
                model_config.set_global8_core_mode()
                logger.info("Using Global8 core mode (all 8 cores)")
            
            # Load model
            self._model = maccel.Model(self.config.model_path, model_config)
            self._model.launch(self._accelerator)
            
            self._initialized = True
            logger.info("Model loaded and launched successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NPU: {str(e)}")
            raise RuntimeError(f"NPU initialization failed: {str(e)}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess image for YOLOv8 inference.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            Tuple of:
                - Preprocessed image ready for inference (N, C, H, W)
                - Metadata dict containing original dimensions and scaling info
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if image.ndim != 3:
            raise ValueError(f"Expected 3D array (H,W,C), got shape {image.shape}")
        
        h0, w0 = image.shape[:2]
        
        # Calculate scaling ratio
        r = min(self.config.img_size[0] / h0, self.config.img_size[1] / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        
        # Calculate padding
        dw = self.config.img_size[1] - new_unpad[0]
        dh = self.config.img_size[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        # Resize image
        import cv2
        if (image.shape[1], image.shape[0]) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Normalize and transpose to CHW
        img = image.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = img[np.newaxis, :]  # Add batch dimension (NCHW)
        
        # Store metadata for coordinate transformation
        metadata = {
            'original_shape': (h0, w0),
            'scale': r,
            'padding': {'top': top, 'left': left},
        }
        
        return img, metadata
    
    def infer(self, preprocessed_image: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on NPU.
        
        Args:
            preprocessed_image: Preprocessed image array (N, C, H, W)
            
        Returns:
            Raw model outputs as list of numpy arrays
            
        Raises:
            RuntimeError: If engine not initialized or inference fails
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        try:
            output = self._model.infer(preprocessed_image)
            return output
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"NPU inference failed: {str(e)}")
    
    # def postprocess(
    #     self, 
    #     raw_outputs: List[np.ndarray], 
    #     metadata: Dict[str, Any]
    # ) -> List[Detection]:
    #     """
    #     Postprocess raw model outputs to detection results.
        
    #     Args:
    #         raw_outputs: Raw model outputs from NPU
    #         metadata: Preprocessing metadata for coordinate transformation
            
    #     Returns:
    #         List of Detection objects with coordinates in original image space
    #     """
    #     if not TORCH_AVAILABLE:
    #         raise RuntimeError("PyTorch is required for postprocessing. Install with: pip install torch")
        
    #     # Process raw outputs using YOLO decoder
    #     detections = self._decode_outputs(raw_outputs)
        
    #     if len(detections) == 0:
    #         return []
        
    #     # Transform coordinates back to original image space
    #     scale = metadata['scale']
    #     padding = metadata['padding']
        
    #     results = []
    #     for det in detections:
    #         x1, y1, x2, y2, conf, cls_id = det
            
    #         # Remove padding offset and scale back
    #         x1 = (x1 - padding['left']) / scale
    #         y1 = (y1 - padding['top']) / scale
    #         x2 = (x2 - padding['left']) / scale
    #         y2 = (y2 - padding['top']) / scale
            
    #         # Clip to image bounds
    #         h0, w0 = metadata['original_shape']
    #         x1 = max(0, min(x1, w0))
    #         y1 = max(0, min(y1, h0))
    #         x2 = max(0, min(x2, w0))
    #         y2 = max(0, min(y2, h0))
            
    #         detection = Detection(
    #             x1=float(x1),
    #             y1=float(y1),
    #             x2=float(x2),
    #             y2=float(y2),
    #             confidence=float(conf),
    #             class_id=int(cls_id),
    #             class_name=self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else None
    #         )
    #         results.append(detection)
        
    #     return results
    # def postprocess(self, raw_outputs, metadata):
    #     if not TORCH_AVAILABLE:
    #         raise RuntimeError("PyTorch is required for postprocessing. Install with: pip install torch")

    #     fmt = self._classify_output_format(raw_outputs)
    #     logger.debug(f"NPU output format: {fmt}")

    #     if fmt.startswith("decoded"):
    #         detections = self._postprocess_decoded_outputs(raw_outputs)
    #     else:
    #         # existing path that builds anchors + DFL projection + NMS
    #         detections = self._decode_outputs(raw_outputs)

    #     if len(detections) == 0:
    #         return []

    #     # === keep your current “remove padding & rescale to original image” block ===
    #     scale  = metadata['scale']
    #     pad    = metadata['padding']
    #     results = []
    #     for det in detections:
    #         x1, y1, x2, y2, conf, cls_id = det
    #         x1 = (x1 - pad['left']) / scale
    #         y1 = (y1 - pad['top'])  / scale
    #         x2 = (x2 - pad['left']) / scale
    #         y2 = (y2 - pad['top'])  / scale

    #         cls_id = int(cls_id)
    #         cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else None
    #         results.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2,
    #                                 confidence=float(conf), class_id=cls_id, class_name=cls_name))
    #     return results

    def postprocess(self, raw_outputs, metadata):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for postprocessing. Install with: pip install torch")

        fmt = self._classify_output_format(raw_outputs)
        logger.debug(f"NPU output format: {fmt}")

        if fmt.startswith("decoded"):
            detections = self._postprocess_decoded_outputs(raw_outputs)
        else:
            # existing path that builds anchors + DFL projection + NMS
            detections = self._decode_outputs(raw_outputs)

        if len(detections) == 0:
            return []

        # === remove padding & rescale to original image ===
        scale = metadata.get('scale', 1.0)
        pad   = metadata.get('padding', {'left': 0.0, 'top': 0.0})
        h0, w0 = metadata.get('original_shape', (None, None))

        results = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det

            # un-letterbox
            x1 = (x1 - pad['left']) / scale
            y1 = (y1 - pad['top'])  / scale
            x2 = (x2 - pad['left']) / scale
            y2 = (y2 - pad['top'])  / scale

            # clip to image bounds (if original size is known)
            if w0 is not None and h0 is not None:
                x1 = max(0.0, min(x1, w0))
                y1 = max(0.0, min(y1, h0))
                x2 = max(0.0, min(x2, w0))
                y2 = max(0.0, min(y2, h0))

            # drop degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(cls_id)
            cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else None
            results.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(conf),
                class_id=cls_id,
                class_name=cls_name
            ))

        return results


    

    def _classify_output_format(self, raw_outputs) -> str:
        """Return 'decoded', 'decoded_per_layer', or 'raw_dfl' based on tensor shapes."""
        chs = []
        for t in raw_outputs:
            # accept numpy or torch; normalize to numpy-like shape
            shape = getattr(t, "shape", None)
            if shape is None or len(shape) < 2:
                continue
            chs.append(int(shape[1]))

        # decoded as a single fused head: [B, 4+nc, N]  → e.g., [B, 21, 8400]
        if len(raw_outputs) == 1 and chs and chs[0] == (4 + self.config.num_classes):
            return "decoded"

        # some runtimes return decoded per layer: 3 tensors, each [B, 4+nc, Hi*Wi]
        if len(raw_outputs) >= 3 and all(c == (4 + self.config.num_classes) for c in chs):
            return "decoded_per_layer"

        # raw DFL: we should see at least one [B, 4*REG_MAX, Hi*Wi] and one [B, nc, Hi*Wi]
        if any(c == self.config.reg_max * 4 for c in chs) and any(c == self.config.num_classes for c in chs):
            return "raw_dfl"

        # fallback
        raise ValueError(f"Unexpected NPU output channels: {chs} "
                        f"(expected {(4+self.config.num_classes)} or {(4*self.config.reg_max)}+{self.config.num_classes})")
    

    def _postprocess_decoded_outputs(self, raw_outputs) -> np.ndarray:
        """
        Handle decoded YOLOv8 heads:
        - Single tensor [B, 4+nc, N]   OR
        - Per-layer tensors [B, 4+nc, Ni] ... (concat along N)
        Returns ndarray of shape [num_det, 6] as [x1,y1,x2,y2,conf,cls] in the *letterboxed* scale.
        """
        import numpy as np
        # Collect into [C, N] (no batch)
        parts = []
        for t in raw_outputs:
            a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            if a.ndim == 3:   # [B,C,N]
                a = a[0]
            elif a.ndim == 2: # [C,N] (some runtimes)
                pass
            else:
                raise ValueError(f"Unexpected decoded tensor shape {a.shape}")
            parts.append(a)

        pred = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)  # [C, sumN]
        C = pred.shape[0]
        assert C >= 4 + self.config.num_classes, f"Decoded channels {C} < 4+nc"

        # Split
        boxes_cxcywh = pred[:4, :]                    # [4, N]
        scores = pred[4:4 + self.config.num_classes]  # [nc, N]

        # If they’re logits, squash; if already [0,1], this is a no-op
        if scores.max() > 1.0 or scores.min() < 0.0:
            scores = 1.0 / (1.0 + np.exp(-scores))

        cls = scores.argmax(axis=0).astype(np.float32)     # [N]
        conf = scores.max(axis=0).astype(np.float32)       # [N]

        # Convert cx,cy,w,h -> x1,y1,x2,y2 in *letterboxed* coordinates (same as your DFL path)
        cx, cy, w, h = boxes_cxcywh
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5

        dets = np.stack([x1, y1, x2, y2, conf, cls], axis=1).astype(np.float32)  # [N, 6]
        # NMS with the same settings as DFL branch
        return self._nms_numpy(dets, self.config.iou_threshold, self.config.conf_threshold)



    def _decode_outputs(self, raw_outputs: List[np.ndarray]) -> List:
        """
        Decode YOLO outputs with anchor generation, DFL decoding, and NMS.
        
        Args:
            raw_outputs: Raw model outputs from NPU
            
        Returns:
            List of detections [x1, y1, x2, y2, conf, cls_id]
        """
        import torch
        import torch.nn.functional as F
        
        device = torch.device("cpu")
        
        # Generate anchors and strides
        anchors, strides = self._make_anchors(
            self.config.num_layers, 
            self.config.img_size
        )
        anchors = anchors.to(device)
        strides = strides.to(device)
        
        # Separate detection and classification outputs
        det_outs = []
        cls_outs = []
        
        for tensor in raw_outputs:
            # Convert to torch tensor if needed
            if not isinstance(tensor, torch.Tensor):
                t = torch.from_numpy(tensor).to(torch.float32)
            else:
                t = tensor
                
            # Add batch dimension if needed
            if t.ndim == 3:
                t = t.unsqueeze(0)
            
            # Classify by channel count
            if t.shape[1] == self.config.reg_max * 4:
                det_outs.append(t)
            elif t.shape[1] == self.config.num_classes:
                cls_outs.append(t)
        
        # Sort by size (largest first)
        det_outs = sorted(det_outs, key=lambda x: x.numel(), reverse=True)
        cls_outs = sorted(cls_outs, key=lambda x: x.numel(), reverse=True)
        
        # Concatenate detection and classification outputs
        outputs = [
            torch.cat((det, cls), dim=1).flatten(2) 
            for det, cls in zip(det_outs, cls_outs)
        ]
        
        # Combine all outputs
        batch_out = torch.cat(outputs, dim=2)[0]
        
        # Split into box and class predictions
        box_raw = batch_out[:self.config.reg_max * 4]
        cls_raw = batch_out[self.config.reg_max * 4:]
        
        # Apply confidence threshold
        scores = cls_raw.max(0)[0]
        threshold = -np.log(1 / self.config.conf_threshold - 1)
        keep = scores > threshold
        
        if keep.sum() == 0:
            return []
        
        # Filter predictions
        box_raw = box_raw[:, keep]
        cls_raw = cls_raw[:, keep]
        anchors_keep = anchors[:, keep]
        strides_keep = strides[:, keep]
        
        # Decode bounding boxes using DFL
        dist = self._decode_dfl(box_raw.unsqueeze(0), self.config.reg_max).squeeze(0)
        boxes = self._dist2bbox(dist.T, anchors_keep.T, xywh=False) * strides_keep.T
        
        # Get class scores and indices
        scores = cls_raw.sigmoid()
        conf, cls_idx = torch.max(scores, dim=0)
        
        # Combine into detections
        dets = torch.cat([
            boxes, 
            conf.unsqueeze(1), 
            cls_idx.float().unsqueeze(1)
        ], dim=1).cpu().numpy()
        
        # Apply NMS
        return self._nms_numpy(dets, self.config.iou_threshold, self.config.conf_threshold)
    
    def _make_anchors(
        self, 
        nl: int = 3, 
        img_size: Tuple[int, int] = (640, 640), 
        offset: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate anchors for YOLO detection.
        
        Args:
            nl: Number of detection layers
            img_size: Image size (h, w)
            offset: Anchor offset
            
        Returns:
            Tuple of anchor points and stride tensors
        """
        import torch
        
        imh, imw = img_size
        anchor_points = []
        stride_tensor = []
        
        # Generate strides for each layer
        strides = [2 ** (3 + i) for i in range(nl)]
        
        for strd in strides:
            ny = imh // strd
            nx = imw // strd
            
            # Create grid
            sy = torch.arange(ny, dtype=torch.float32) + offset
            sx = torch.arange(nx, dtype=torch.float32) + offset
            yv, xv = torch.meshgrid(sy, sx, indexing="ij")
            
            # Stack and reshape
            anchor_points.append(torch.stack((xv, yv), -1).reshape(-1, 2))
            stride_tensor.append(torch.full((ny * nx, 1), strd, dtype=torch.float32))
        
        # Concatenate all anchors and strides
        anchors = torch.cat(anchor_points, dim=0).T
        strides = torch.cat(stride_tensor, dim=0).T
        
        return anchors, strides
    
    def _decode_dfl(self, x: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
        """
        Decode Distribution Focal Loss (DFL) predictions.
        
        Args:
            x: Input tensor
            reg_max: Maximum regression value
            
        Returns:
            Decoded tensor
        """
        import torch
        import torch.nn.functional as F
        
        b, _, a = x.shape
        
        # Create DFL weights
        dfl_weight = torch.arange(reg_max, dtype=torch.float).reshape(1, -1, 1, 1)
        
        # Reshape and apply softmax
        x = x.view(b, 4, reg_max, a).transpose(2, 1).softmax(1)
        
        # Apply convolution with DFL weights
        return F.conv2d(x, dfl_weight).view(b, 4, a)
    
    def _dist2bbox(
        self, 
        distance: torch.Tensor, 
        anchor_points: torch.Tensor, 
        xywh: bool = True, 
        dim: int = -1
    ) -> torch.Tensor:
        """
        Convert distance predictions to bounding boxes.
        
        Args:
            distance: Distance tensor
            anchor_points: Anchor points
            xywh: Return as xywh format (True) or xyxy format (False)
            dim: Dimension for splitting
            
        Returns:
            Bounding box tensor
        """
        import torch
        
        # Split distance into left-top and right-bottom
        lt, rb = distance.chunk(2, dim)
        
        # Calculate corners
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        
        if xywh:
            # Return as center + width/height
            return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
        else:
            # Return as corners
            return torch.cat((x1y1, x2y2), dim)
    


    def _nms_numpy(self, dets, iou_thresh: float = 0.45, conf_thresh: float = 0.3, topk: int = 300, class_aware: bool = True):

        import numpy as np
        if len(dets) == 0:
            return []
        dets = np.asarray(dets, dtype=np.float32)
        x1, y1, x2, y2, cf, cl = dets.T
        order = cf.argsort()[::-1]
        areas = (x2 - x1) * (y2 - y1)
        keep = []
        while order.size > 0:
            i = order[0]
            if cf[i] < conf_thresh:
                break  # enforce confidence threshold
            keep.append(dets[i])
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            if class_aware:
                same_cls = (cl[order[1:]] == cl[i])
                keep_mask = ~((iou > iou_thresh) & same_cls)
            else:
                keep_mask = ~(iou > iou_thresh)
            order = order[1:][keep_mask]
            if len(keep) >= topk:
                break
        return keep

    
    # def _nms_numpy(
    #     self, 
    #     dets: np.ndarray, 
    #     iou_thresh: float = 0.45, 
    #     conf_thresh: float = 0.3
    # ) -> List:
    #     """
    #     Non-Maximum Suppression using NumPy.
        
    #     Args:
    #         dets: Detection array [x1, y1, x2, y2, conf, cls]
    #         iou_thresh: IOU threshold for NMS
    #         conf_thresh: Confidence threshold
            
    #     Returns:
    #         List of filtered detections
    #     """
    #     if len(dets) == 0:
    #         return []
        
    #     # Extract components
    #     x1 = dets[:, 0]
    #     y1 = dets[:, 1]
    #     x2 = dets[:, 2]
    #     y2 = dets[:, 3]
    #     conf = dets[:, 4]
    #     cls = dets[:, 5]
        
    #     # Calculate areas
    #     areas = (x2 - x1) * (y2 - y1)
        
    #     # Sort by confidence
    #     order = conf.argsort()[::-1]
        
    #     keep = []
    #     while order.size > 0:
    #         i = order[0]
    #         keep.append(dets[i])
            
    #         if order.size == 1:
    #             break
            
    #         # Calculate IoU with remaining boxes
    #         xx1 = np.maximum(x1[i], x1[order[1:]])
    #         yy1 = np.maximum(y1[i], y1[order[1:]])
    #         xx2 = np.minimum(x2[i], x2[order[1:]])
    #         yy2 = np.minimum(y2[i], y2[order[1:]])
            
    #         # Calculate intersection area
    #         inter_w = np.maximum(0.0, xx2 - xx1)
    #         inter_h = np.maximum(0.0, yy2 - yy1)
    #         inter = inter_w * inter_h
            
    #         # Calculate IoU
    #         iou = inter / (areas[i] + areas[order[1:]] - inter)
            
    #         # Keep boxes with IoU less than threshold
    #         inds = np.where(iou <= iou_thresh)[0]
    #         order = order[inds + 1]
            
    #         # Limit maximum detections
    #         if len(keep) >= 300:
    #             break
        
    #     return keep
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        End-to-end detection pipeline.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            List of Detection objects
        """
        # Preprocess
        preprocessed, metadata = self.preprocess(image)
        
        # Inference
        raw_outputs = self.infer(preprocessed)
        
        # Postprocess
        detections = self.postprocess(raw_outputs, metadata)
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Process multiple images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of detection lists (one per image)
        """
        results = []
        for image in images:
            detections = self.detect(image)
            results.append(detections)
        return results
    
    def cleanup(self) -> None:
        """Clean up NPU resources."""
        if self._model:
            try:
                self._model.dispose()
                logger.info("Model disposed successfully")
            except Exception as e:
                logger.error(f"Error disposing model: {str(e)}")
        
        self._model = connection = None
        self._accelerator = None
        self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get inference engine statistics.
        
        Returns:
            Dictionary containing engine statistics
        """
        return {
            'initialized': self._initialized,
            'model_path': self.config.model_path,
            'device_id': self.config.device_id,
            'img_size': self.config.img_size,
            'num_classes': self.config.num_classes,
            'conf_threshold': self.config.conf_threshold,
            'iou_threshold': self.config.iou_threshold,
        }