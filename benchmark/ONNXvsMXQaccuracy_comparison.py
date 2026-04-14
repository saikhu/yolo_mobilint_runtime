#!/usr/bin/env python3
"""
Fixed ONNX (GPU) vs MXQ (NPU) Accuracy Comparison Script
Proper postprocessing and timing collection
"""

import os
import cv2
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import existing modules
from yolo_mobilint_runtime.src.npu_inference_yolov8 import NPUInferenceEngine, InferenceConfig, Detection
from accuracy_validator import AccuracyValidator, AccuracyResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComparisonConfig:
    """Configuration for ONNX vs MXQ comparison"""
    # Model paths
    npu_model_path: str
    onnx_model_path: str
    
    # Test configuration
    num_test_frames: int = 100
    img_size: Tuple[int, int] = (640, 640)
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    
    # Output configuration
    output_dir: str = "accuracy_comparison"
    save_images: bool = True
    create_plots: bool = True

class FixedONNXEngine:
    """Fixed ONNX inference engine with proper YOLOv8 postprocessing"""
    
    def __init__(self, model_path: str, img_size=(640, 640), conf_threshold=0.5, iou_threshold=0.5):
        self.model_path = model_path
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.input_name = None
        self.output_names = None
        self.initialized = False
        
        # Performance tracking
        from collections import deque
        self.frame_times = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        
        # Class names matching NPU setup
        self.class_names = [
            "backhoe_loader", "cement_truck", "compactor", "dozer", "dump_truck",
            "excavator", "grader", "mobile_crane", "tower_crane", "wheel_loader",
            "worker", "Hardhat", "Red_Hardhat", "scaffolds", "Lifted Load",
            "Crane_Hook", "Hook"
        ]
    
    def initialize(self):
        """Initialize ONNX model"""
        try:
            import onnxruntime as ort
            
            # Set providers
            providers = []
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            
            # Load model
            self.model = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output info
            self.input_name = self.model.get_inputs()[0].name
            self.output_names = [output.name for output in self.model.get_outputs()]
            
            # Print model info for debugging
            input_shape = self.model.get_inputs()[0].shape
            logger.info(f"ONNX model input shape: {input_shape}")
            logger.info(f"ONNX model outputs: {len(self.output_names)}")
            for i, output in enumerate(self.model.get_outputs()):
                logger.info(f"  Output {i}: {output.name}, shape: {output.shape}")
            
            self.initialized = True
            logger.info(f"ONNX model initialized: {self.model_path}")
            
        except ImportError:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX model: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Preprocess image for ONNX inference - match NPU preprocessing exactly"""
        h0, w0 = image.shape[:2]
        
        # Calculate scaling ratio (same as NPU)
        r = min(self.img_size[0] / h0, self.img_size[1] / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        
        # Calculate padding
        dw = self.img_size[1] - new_unpad[0]
        dh = self.img_size[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        # Resize image
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
    
    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], Dict[str, float]]:
        """Run ONNX inference and return detections"""
        if not self.initialized:
            raise RuntimeError("Engine not initialized")
        
        t0 = time.perf_counter()
        
        # Preprocess
        preprocessed, metadata = self.preprocess(frame)
        t1 = time.perf_counter()
        
        # Inference
        outputs = self.model.run(self.output_names, {self.input_name: preprocessed})
        t2 = time.perf_counter()
        
        # Postprocess - use proper YOLOv8 postprocessing
        detections = self._postprocess_yolov8_outputs(outputs, metadata)
        t3 = time.perf_counter()
        
        # Track timing
        inference_time = (t2 - t1) * 1000
        total_time = (t3 - t0) * 1000
        
        self.inference_times.append(inference_time)
        self.frame_times.append(total_time)
        
        timing = {
            'preprocess_ms': (t1 - t0) * 1000,
            'inference_ms': inference_time,
            'postprocess_ms': (t3 - t2) * 1000,
            'total_ms': total_time
        }
        
        return detections, timing
    def _nms_numpy(self, dets: np.ndarray, iou_thr: float = 0.45, conf_thr: float = 0.25, topk: int = 300) -> np.ndarray:
        """
        Class-aware NMS on dets = [x1,y1,x2,y2,conf,cls]
        Returns filtered array with same columns.
        """
        if dets.size == 0:
            return dets
        x1, y1, x2, y2, cf, cl = dets.T
        order = cf.argsort()[::-1]
        areas = (x2 - x1) * (y2 - y1)
        keep_idx = []
        while order.size > 0:
            i = order[0]
            if cf[i] < conf_thr:
                break
            keep_idx.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter + 1e-9
            iou = inter / union
            same_cls = (cl[order[1:]] == cl[i])
            suppress = (iou > iou_thr) & same_cls  # suppress same-class overlaps
            remain = np.where(~suppress)[0]
            order = order[remain + 1]
            if len(keep_idx) >= topk:
                break
        return dets[keep_idx] if keep_idx else dets[:0]

    
    def _unletterbox_xyxy(self, xyxy: np.ndarray, metadata: Dict) -> np.ndarray:
        """Map [x1,y1,x2,y2] from letterboxed input back to ORIGINAL image coords."""
        if xyxy.size == 0:
            return xyxy
        xyxy = xyxy.copy()
        padL = float(metadata['padding']['left'])
        padT = float(metadata['padding']['top'])
        s = float(metadata['scale'])
        h0, w0 = metadata['original_shape']
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - padL) / s
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - padT) / s
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, float(w0))
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, float(h0))
        return xyxy

    def _postprocess_yolov8_outputs(self, outputs, metadata):
        """
        Robust postprocess for common YOLOv8 ONNX export variants:
        - NMS included:        [num,6] or [1,num,6]  -> (x1,y1,x2,y2,conf,cls)
        - Decoded (no NMS):    [1, 4+nc, N] or [1, N, 4+nc]  (cx,cy,w,h + class logits/scores)
        - (If raw DFL heads are ever returned, add a DFL branch like your NPU.)
        Uses self.conf_threshold and self.iou_threshold. Returns List[Detection].
        """
        detections = []
        if not outputs:
            return detections

        pred = np.array(outputs[0])  # first (and usually only) output
        logger.debug(f"ONNX raw output shape: {pred.shape!r}")

        # --- Case A: NMS already applied: [num,6] or [1,num,6] (x1,y1,x2,y2,conf,cls) ---
        arr = pred
        if arr.ndim == 3 and arr.shape[-1] in (6, 7):  # [1,num,6]
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[-1] in (6, 7):
            logger.debug("Postprocess branch: NMS-included outputs")
            boxes = arr[:, :6].astype(np.float32, copy=True)  # x1,y1,x2,y2,conf,cls
            # Map from letterboxed to original coords
            if boxes.size:
                boxes[:, :4] = self._unletterbox_xyxy(boxes[:, :4], metadata)
                # Confidence filter (defensive)
                boxes = boxes[boxes[:, 4] >= float(self.conf_threshold)]
            for x1, y1, x2, y2, conf, clsid in boxes:
                clsid = int(clsid)
                cname = self.class_names[clsid] if 0 <= clsid < len(self.class_names) else None
                detections.append(Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    confidence=float(conf), class_id=clsid, class_name=cname
                ))
            return detections

        # --- Case B: Decoded predictions (no NMS): [1, 4+nc, N] or [1, N, 4+nc] ---
        if pred.ndim == 3:
            arr = pred[0]  # drop batch: shape now either [4+nc, N] or [N, 4+nc]
            if arr.shape[0] == 4 + self.num_classes:  # [4+nc, N]
                boxes_cxcywh = arr[:4, :]
                scores = arr[4:4 + self.num_classes, :]
            elif arr.shape[1] == 4 + self.num_classes:  # [N, 4+nc]
                arr = arr.T
                boxes_cxcywh = arr[:4, :]
                scores = arr[4:4 + self.num_classes, :]
            else:
                logger.warning(f"Unexpected decoded ONNX shape after batch drop: {pred.shape} -> {arr.shape}")
                return detections

            # If these look like logits, apply sigmoid
            if scores.max() > 1.0 or scores.min() < 0.0:
                scores = 1.0 / (1.0 + np.exp(-scores))

            # Build candidate boxes in ORIGINAL image coords
            cand = []
            N = arr.shape[1]
            for i in range(N):
                clsid = int(np.argmax(scores[:, i]))
                conf = float(scores[clsid, i])
                if conf < float(self.conf_threshold):
                    continue
                xc, yc, w, h = boxes_cxcywh[:, i]
                x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                x1, y1, x2, y2 = self._transform_coordinates(x1, y1, x2, y2, metadata)  # to original space
                cand.append([x1, y1, x2, y2, conf, float(clsid)])

            if not cand:
                return detections

            cand = np.asarray(cand, dtype=np.float32)
            kept = self._nms_numpy(cand, iou_thr=float(self.iou_threshold), conf_thr=float(self.conf_threshold), topk=300)
            for x1, y1, x2, y2, conf, clsid in kept:
                clsid = int(clsid)
                cname = self.class_names[clsid] if 0 <= clsid < len(self.class_names) else None
                detections.append(Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    confidence=float(conf), class_id=clsid, class_name=cname
                ))
            return detections

        logger.warning(f"Unsupported ONNX output shape: {pred.shape}. Returning no detections.")
        return detections


    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to detections"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            # Take the detection with highest confidence
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                iou = self._calculate_iou(
                    [current.x1, current.y1, current.x2, current.y2],
                    [det.x1, det.y1, det.x2, det.y2]
                )
                
                # Keep only if IoU is below threshold and different class
                if iou <= self.iou_threshold or current.class_id != det.class_id:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _transform_coordinates(self, x1, y1, x2, y2, metadata):
        """Transform coordinates back to original image space"""
        scale = metadata['scale']
        padding = metadata['padding']
        
        # Remove padding offset and scale back
        x1 = (x1 - padding['left']) / scale
        y1 = (y1 - padding['top']) / scale
        x2 = (x2 - padding['left']) / scale
        y2 = (y2 - padding['top']) / scale
        
        # Clip to image bounds
        h0, w0 = metadata['original_shape']
        x1 = max(0, min(x1, w0))
        y1 = max(0, min(y1, h0))
        x2 = max(0, min(x2, w0))
        y2 = max(0, min(y2, h0))
        
        return x1, y1, x2, y2
    
    @property
    def num_classes(self):
        return len(self.class_names)
    
    def cleanup(self):
        """Clean up resources"""
        self.model = None
        self.initialized = False

class TimingWrapper:
    """Wrapper to capture NPU timing properly"""
    
    def __init__(self, npu_engine):
        self.npu_engine = npu_engine
        self.frame_times = []
    
    def detect(self, frame):
        """Detect with timing capture"""
        t0 = time.perf_counter()
        detections = self.npu_engine.detect(frame)
        t1 = time.perf_counter()
        
        frame_time = (t1 - t0) * 1000
        self.frame_times.append(frame_time)
        
        return detections

class FixedAccuracyComparison:
    """Fixed accuracy comparison with proper timing and postprocessing"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.npu_engine = None
        self.onnx_engine = None
        self.npu_wrapper = None
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def initialize_engines(self):
        """Initialize both NPU and ONNX engines"""
        logger.info("Initializing engines...")
        
        # Initialize NPU engine
        try:
            npu_config = InferenceConfig(
                model_path=self.config.npu_model_path,
                img_size=self.config.img_size,
                conf_threshold=self.config.conf_threshold,
                iou_threshold=self.config.iou_threshold,
                use_global8_core=True
            )
            self.npu_engine = NPUInferenceEngine(npu_config)
            self.npu_engine.initialize()
            self.npu_wrapper = TimingWrapper(self.npu_engine)
            logger.info("NPU engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NPU engine: {e}")
            raise
        
        # Initialize ONNX engine
        try:
            self.onnx_engine = FixedONNXEngine(
                self.config.onnx_model_path,
                self.config.img_size,
                self.config.conf_threshold,
                self.config.iou_threshold
            )
            self.onnx_engine.initialize()
            logger.info("ONNX engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX engine: {e}")
            raise
    
    def load_test_frames(self, video_path: str = None) -> List[np.ndarray]:
        """Load test frames from video or create synthetic frames"""
        frames = []
        
        if video_path and os.path.exists(video_path):
            logger.info(f"Loading frames from video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            while len(frames) < self.config.num_test_frames and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Take every 10th frame to get variety
                if frame_count % 10 == 0:
                    frames.append(frame.copy())
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Loaded {len(frames)} frames from video")
        
        else:
            # Create synthetic test frames
            logger.info("Creating synthetic test frames")
            for i in range(self.config.num_test_frames):
                # Create random image with some patterns
                img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
                
                # Add some shapes to make it more realistic
                cv2.rectangle(img, (100 + i*5, 100 + i*3), (200 + i*5, 200 + i*3), 
                            (255, 0, 0), -1)
                cv2.circle(img, (300 + i*2, 300 + i*2), 50, (0, 255, 0), -1)
                
                frames.append(img)
        
        return frames
    
    def run_accuracy_comparison(self, video_path: str = None) -> Dict:
        """Run comprehensive accuracy comparison"""
        logger.info("Starting accuracy comparison...")
        
        # Initialize engines
        self.initialize_engines()
        
        # Load test frames
        test_frames = self.load_test_frames(video_path)
        
        if len(test_frames) == 0:
            raise RuntimeError("No test frames available")
        
        # Run inference on all frames
        npu_results = []
        onnx_results = []
        
        logger.info(f"Running inference on {len(test_frames)} frames...")
        
        for i, frame in enumerate(test_frames):
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_frames)} frames")
            
            # NPU inference with timing
            try:
                npu_detections = self.npu_wrapper.detect(frame)
                npu_results.append(npu_detections)
            except Exception as e:
                logger.warning(f"NPU inference failed on frame {i}: {e}")
                npu_results.append([])
            
            # ONNX inference
            try:
                onnx_detections, _ = self.onnx_engine.infer(frame)
                onnx_results.append(onnx_detections)
            except Exception as e:
                logger.warning(f"ONNX inference failed on frame {i}: {e}")
                onnx_results.append([])
        
        # Calculate accuracy and performance metrics
        accuracy_results = self._calculate_accuracy_metrics(npu_results, onnx_results)
        performance_results = self._calculate_performance_metrics()
        
        # Save results
        results = {
            'accuracy': accuracy_results,
            'performance': performance_results,
            'config': asdict(self.config)
        }
        
        self._save_results(results)
        
        if self.config.create_plots:
            self._create_comparison_plots(results)
        
        return results
    
    def _calculate_accuracy_metrics(self, npu_results, onnx_results) -> Dict:
        """Calculate accuracy metrics between NPU and ONNX results"""
        total_matches = 0
        total_close_matches = 0
        total_npu_detections = 0
        total_onnx_detections = 0
        all_ious = []
        all_conf_diffs = []
        
        validator = AccuracyValidator(
            iou_threshold=self.config.iou_threshold,
            conf_threshold=self.config.conf_threshold
        )
        
        for npu_dets, onnx_dets in zip(npu_results, onnx_results):
            if not npu_dets and not onnx_dets:
                continue
            
            # Match detections
            match_result = validator.match_detections(npu_dets, onnx_dets)
            
            # Update statistics
            matches = match_result['matches']
            total_matches += len([m for m in matches if m['iou'] > 0.9])
            total_close_matches += len(matches)
            
            total_npu_detections += len(npu_dets)
            total_onnx_detections += len(onnx_dets)
            
            for match in matches:
                all_ious.append(match['iou'])
                all_conf_diffs.append(match['conf_diff'])
        
        return {
            'total_frames': len(npu_results),
            'total_npu_detections': total_npu_detections,
            'total_onnx_detections': total_onnx_detections,
            'exact_matches': total_matches,
            'close_matches': total_close_matches,
            'avg_iou': np.mean(all_ious) if all_ious else 0.0,
            'avg_conf_diff': np.mean(all_conf_diffs) if all_conf_diffs else 0.0,
            'detection_agreement_rate': total_close_matches / max(total_npu_detections, total_onnx_detections, 1) if max(total_npu_detections, total_onnx_detections) > 0 else 0.0,
        }
    

    def _calculate_performance_metrics(self) -> Dict:
        """
        Keep original keys for compatibility, but:
        - npu_avg_inference_ms is derived from total frame times (no split available)
        - onnx_avg_inference_ms uses true infer-only timings
        - also add *_avg_total_ms for clarity
        """
        npu_total = list(self.npu_wrapper.frame_times) if self.npu_wrapper else []
        onnx_total = list(self.onnx_engine.frame_times) if self.onnx_engine else []
        onnx_infer = list(self.onnx_engine.inference_times) if self.onnx_engine else []

        npu_avg_total_ms = float(np.mean(npu_total)) if npu_total else 0.0
        onnx_avg_total_ms = float(np.mean(onnx_total)) if onnx_total else 0.0
        onnx_avg_infer_ms = float(np.mean(onnx_infer)) if onnx_infer else 0.0

        return {
            # kept for backward compatibility (note: NPU = total)
            'npu_avg_inference_ms': npu_avg_total_ms,
            'onnx_avg_inference_ms': onnx_avg_infer_ms,

            # added for clarity
            'npu_avg_total_ms': npu_avg_total_ms,
            'onnx_avg_total_ms': onnx_avg_total_ms,

            'npu_avg_fps': (1000.0 / npu_avg_total_ms) if npu_avg_total_ms > 0 else 0.0,
            'onnx_avg_fps': (1000.0 / onnx_avg_total_ms) if onnx_avg_total_ms > 0 else 0.0,
            'total_frames_processed': len(npu_total)
        }

    
    def _save_results(self, results: Dict):
        """Save comparison results to files"""
        # Save as JSON
        json_path = Path(self.config.output_dir) / "ONNXvsMXQaccuracy_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {json_path}")
        
        # Save detailed report
        self._generate_detailed_report(results)
    
    def _generate_detailed_report(self, results: Dict):
        """Generate detailed text report"""
        report_path = Path(self.config.output_dir) / "ONNXvsMXQaccuracy_comparison.txt"
        
        with open(report_path, 'w') as f:
            f.write("Fixed ONNX vs MXQ Accuracy Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Configuration
            f.write("Configuration:\n")
            f.write(f"  NPU Model: {self.config.npu_model_path}\n")
            f.write(f"  ONNX Model: {self.config.onnx_model_path}\n")
            f.write(f"  Test Frames: {self.config.num_test_frames}\n")
            f.write(f"  Confidence Threshold: {self.config.conf_threshold}\n")
            f.write(f"  IoU Threshold: {self.config.iou_threshold}\n\n")
            
            # Accuracy Results
            acc = results['accuracy']
            f.write("Accuracy Results:\n")
            f.write(f"  Total Frames: {acc['total_frames']}\n")
            f.write(f"  NPU Total Detections: {acc['total_npu_detections']}\n")
            f.write(f"  ONNX Total Detections: {acc['total_onnx_detections']}\n")
            f.write(f"  Exact Matches (IoU > 0.9): {acc['exact_matches']}\n")
            f.write(f"  Close Matches (IoU > {self.config.iou_threshold}): {acc['close_matches']}\n")
            f.write(f"  Average IoU: {acc['avg_iou']:.3f}\n")
            f.write(f"  Average Confidence Diff: {acc['avg_conf_diff']:.3f}\n")
            f.write(f"  Detection Agreement Rate: {acc['detection_agreement_rate']:.3f}\n\n")
            
            # Performance Results
            perf = results['performance']
            f.write("Performance Results:\n")
            f.write(f"  NPU Average Inference: {perf['npu_avg_inference_ms']:.2f} ms\n")
            f.write(f"  ONNX Average Inference: {perf['onnx_avg_inference_ms']:.2f} ms\n")
            f.write(f"  NPU Average FPS: {perf['npu_avg_fps']:.2f}\n")
            f.write(f"  ONNX Average FPS: {perf['onnx_avg_fps']:.2f}\n")
        
        logger.info(f"Detailed report saved to {report_path}")
    
    def _create_comparison_plots(self, results: Dict):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        acc = results['accuracy']
        perf = results['performance']
        
        # Accuracy comparison
        if acc['total_npu_detections'] > 0:
            labels = ['Exact Matches', 'Close Matches', 'No Match']
            sizes = [
                acc['exact_matches'], 
                acc['close_matches'] - acc['exact_matches'], 
                acc['total_npu_detections'] - acc['close_matches']
            ]
            axes[0,0].pie([max(0, s) for s in sizes], labels=labels, autopct='%1.1f%%')
            axes[0,0].set_title('Detection Matching Results')
        else:
            axes[0,0].text(0.5, 0.5, 'No detections found', ha='center', va='center')
            axes[0,0].set_title('Detection Matching Results')
        
        # Performance comparison
        devices = ['NPU (MXQ)', 'ONNX (GPU)']
        fps_values = [perf['npu_avg_fps'], perf['onnx_avg_fps']]
        axes[0,1].bar(devices, fps_values)
        axes[0,1].set_title('Average FPS Comparison')
        axes[0,1].set_ylabel('FPS')
        
        # Inference time comparison
        latency_values = [perf['npu_avg_inference_ms'], perf['onnx_avg_inference_ms']]
        axes[1,0].bar(devices, latency_values)
        axes[1,0].set_title('Average Inference Time')
        axes[1,0].set_ylabel('Milliseconds')
        
        # Detection count comparison
        det_counts = [acc['total_npu_detections'], acc['total_onnx_detections']]
        total_frames = acc['total_frames']
        if total_frames > 0:
            det_per_frame = [d/total_frames for d in det_counts]
            axes[1,1].bar(devices, det_per_frame)
            axes[1,1].set_title('Average Detections per Frame')
            axes[1,1].set_ylabel('Count')
        else:
            axes[1,1].text(0.5, 0.5, 'No frames processed', ha='center', va='center')
        
        plt.tight_layout()
        plot_path = Path(self.config.output_dir) / "fixed_comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to {plot_path}")
        plt.close()
    
    def cleanup(self):
        """Clean up resources"""
        if self.npu_engine:
            self.npu_engine.cleanup()
        if self.onnx_engine:
            self.onnx_engine.cleanup()

def main():
    """Main execution function"""
    # Configuration
    config = ComparisonConfig(
        npu_model_path="../models/yolov8s_best_globalCore_v1.mxq",
        onnx_model_path="../models/yolov8s_best.onnx",
        num_test_frames=10000,  # Reduced for faster testing
        output_dir="ONNXvsMXQaccuracy_comparison_v2"
    )
    
    print("Fixed ONNX vs MXQ Accuracy Comparison")
    print("=" * 45)
    print(f"NPU Model: {config.npu_model_path}")
    print(f"ONNX Model: {config.onnx_model_path}")
    print(f"Test Frames: {config.num_test_frames}")
    print("=" * 45)
    
    # Check if model files exist
    if not os.path.exists(config.npu_model_path):
        print(f"Error: NPU model not found at {config.npu_model_path}")
        return
    
    if not os.path.exists(config.onnx_model_path):
        print(f"Error: ONNX model not found at {config.onnx_model_path}")
        print("Please convert your model to ONNX format or update the path")
        return
    

    # Find test video
    test_video = None
    for video_path in ["input/0.mp4", "input/1.mp4", "input/2.mp4, input/3.mp4", "input/4.mp4",
                       "input/0.mp4", "input/1.mp4", "input/2.mp4, input/3.mp4", "input/4.mp4",
                       "input/0.mp4", "input/1.mp4", "input/2.mp4, input/3.mp4", "input/4.mp4",
                       "input/0.mp4", "input/1.mp4", "input/2.mp4, input/3.mp4", "input/4.mp4"]:
        if os.path.exists(video_path):
            test_video = video_path
            break
    
    if test_video:
        print(f"Using test video: {test_video}")
    else:
        print("No test video found, using synthetic frames")
    
    # THIS IS THE MISSING PART - Actually run the comparison
    comparison = FixedAccuracyComparison(config)
    
    try:
        results = comparison.run_accuracy_comparison(test_video)
        
        # Print results
        acc = results['accuracy']
        perf = results['performance']
        
        print(f"\nResults:")
        print(f"  NPU Detections: {acc['total_npu_detections']}")
        print(f"  ONNX Detections: {acc['total_onnx_detections']}")
        print(f"  Agreement Rate: {acc['detection_agreement_rate']:.1%}")
        print(f"  NPU FPS: {perf['npu_avg_fps']:.1f}")
        print(f"  ONNX FPS: {perf['onnx_avg_fps']:.1f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        comparison.cleanup()

print("Script started")


if __name__ == "__main__":
    print("About to call main")
    main()