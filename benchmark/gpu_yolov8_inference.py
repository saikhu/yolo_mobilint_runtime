"""
GPU YOLOv8 Inference Implementation
Compatible interface with NPU inference for fair comparison
"""

import cv2
import numpy as np
import torch
import time
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)

class GPUYOLOv8Engine:
    """
    GPU inference engine for YOLOv8 using PyTorch or ONNX
    Designed to match the NPU inference interface
    """
    
    def __init__(self, model_path: str, device: str = "cuda", engine_type: str = "ultralytics"):
        """
        Initialize GPU inference engine
        
        Args:
            model_path: Path to model file (.pt, .onnx, .torchscript)
            device: Device to use ('cuda' or 'cpu')
            engine_type: 'ultralytics', 'onnx', or 'torchscript'
        """
        self.model_path = model_path
        self.device = device
        self.engine_type = engine_type
        self.model = None
        self.img_size = (640, 640)
        self.conf_threshold = 0.5
        self.iou_threshold = 0.5
        
        # Class names matching your NPU setup
        self.class_names = [
            "backhoe_loader", "cement_truck", "compactor", "dozer", "dump_truck",
            "excavator", "grader", "mobile_crane", "tower_crane", "wheel_loader",
            "worker", "Hardhat", "Red_Hardhat", "scaffolds", "Lifted Load",
            "Crane_Hook", "Hook"
        ]
        
        # Performance tracking
        self.frame_times = deque(maxlen=1000)
        self.preprocess_times = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        self.postprocess_times = deque(maxlen=1000)
        
        self._initialized = False
    
    def initialize(self):
        """Initialize the GPU model"""
        try:
            if self.engine_type == "ultralytics":
                self._init_ultralytics()
            elif self.engine_type == "onnx":
                self._init_onnx()
            elif self.engine_type == "torchscript":
                self._init_torchscript()
            else:
                raise ValueError(f"Unsupported engine type: {self.engine_type}")
            
            self._initialized = True
            logger.info(f"GPU engine initialized: {self.engine_type} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU engine: {e}")
            raise
    
    def _init_ultralytics(self):
        """Initialize using Ultralytics YOLOv8"""
        from ultralytics import YOLO
        
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        
        # Warm up
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy_img, verbose=False)
    
    def _init_onnx(self):
        """Initialize using ONNX Runtime"""
        import onnxruntime as ort
        
        providers = []
        if self.device == "cuda" and ort.get_available_providers():
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        self.model = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get input/output info
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [output.name for output in self.model.get_outputs()]
    
    def _init_torchscript(self):
        """Initialize using TorchScript"""
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()
        
        # Warm up
        dummy_input = torch.randn(1, 3, 640, 640, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
    
    def preprocess(self, image: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """
        Preprocess image for inference
        Should match NPU preprocessing as closely as possible
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be numpy array")
        
        h0, w0 = image.shape[:2]
        
        # Calculate scaling (same as NPU)
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
        
        # Store metadata for coordinate transformation
        metadata = {
            'original_shape': (h0, w0),
            'scale': r,
            'padding': {'top': top, 'left': left},
        }
        
        if self.engine_type == "ultralytics":
            # For Ultralytics, return the image directly
            return image, metadata
        else:
            # For ONNX/TorchScript, normalize and convert to tensor
            img = image.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            
            if self.engine_type == "onnx":
                img = img[np.newaxis, :]  # Add batch dimension for ONNX
                return img, metadata
            else:  # TorchScript
                img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
                return img_tensor, metadata
    
    def infer(self, image: np.ndarray) -> Tuple[List, Dict[str, float]]:
        """
        Run inference and return detections with timing info
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        t0 = time.perf_counter()
        
        # Preprocessing
        preprocessed, metadata = self.preprocess(image)
        t1 = time.perf_counter()
        
        # Inference
        if self.engine_type == "ultralytics":
            raw_outputs = self._infer_ultralytics(preprocessed)
        elif self.engine_type == "onnx":
            raw_outputs = self._infer_onnx(preprocessed)
        else:  # TorchScript
            raw_outputs = self._infer_torchscript(preprocessed)
        
        t2 = time.perf_counter()
        
        # Postprocessing
        detections = self.postprocess(raw_outputs, metadata)
        t3 = time.perf_counter()
        
        # Track timing
        preprocess_time = (t1 - t0) * 1000
        inference_time = (t2 - t1) * 1000
        postprocess_time = (t3 - t2) * 1000
        total_time = (t3 - t0) * 1000
        
        self.preprocess_times.append(preprocess_time)
        self.inference_times.append(inference_time)
        self.postprocess_times.append(postprocess_time)
        self.frame_times.append(total_time)
        
        timing = {
            'preprocess_ms': preprocess_time,
            'inference_ms': inference_time,
            'postprocess_ms': postprocess_time,
            'total_ms': total_time
        }
        
        return detections, timing
    
    def _infer_ultralytics(self, image):
        """Inference using Ultralytics YOLO"""
        results = self.model.predict(
            image, 
            verbose=False, 
            conf=self.conf_threshold,
            iou=self.iou_threshold
        )
        return results[0] if results else None
    
    def _infer_onnx(self, image_tensor):
        """Inference using ONNX Runtime"""
        outputs = self.model.run(self.output_names, {self.input_name: image_tensor})
        return outputs
    
    def _infer_torchscript(self, image_tensor):
        """Inference using TorchScript"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
        return outputs
    
    def postprocess(self, raw_outputs, metadata) -> List:
        """
        Postprocess model outputs to detection format
        """
        detections = []
        
        if self.engine_type == "ultralytics":
            if raw_outputs and raw_outputs.boxes is not None:
                boxes = raw_outputs.boxes
                
                for i in range(len(boxes.xyxy)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Transform coordinates back to original image space
                    x1, y1, x2, y2 = self._transform_coordinates(
                        x1, y1, x2, y2, metadata
                    )
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else None
                    }
                    detections.append(detection)
        
        else:
            # For ONNX/TorchScript, implement custom postprocessing
            # This would depend on your specific model output format
            detections = self._postprocess_raw_outputs(raw_outputs, metadata)
        
        return detections
    
    def _postprocess_raw_outputs(self, raw_outputs, metadata):
        """
        Postprocess raw model outputs (for ONNX/TorchScript)
        This is simplified - you'd need to implement full YOLO postprocessing
        """
        detections = []
        
        # Placeholder implementation
        # You would need to implement:
        # 1. Decode YOLO outputs (similar to your NPU postprocessing)
        # 2. Apply NMS
        # 3. Transform coordinates
        
        return detections
    
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
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'avg_fps': 1000.0 / np.mean(self.frame_times) if self.frame_times else 0,
            'avg_preprocess_ms': np.mean(self.preprocess_times) if self.preprocess_times else 0,
            'avg_inference_ms': np.mean(self.inference_times) if self.inference_times else 0,
            'avg_postprocess_ms': np.mean(self.postprocess_times) if self.postprocess_times else 0,
            'avg_total_ms': np.mean(self.frame_times) if self.frame_times else 0,
        }
    
    def detect(self, image: np.ndarray) -> List:
        """
        End-to-end detection (compatible with NPU interface)
        """
        detections, _ = self.infer(image)
        return detections
    
    def cleanup(self):
        """Clean up GPU resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self._initialized = False
        logger.info("GPU engine cleaned up")

# Benchmark-specific GPU engine wrapper
class BenchmarkGPUEngine:
    """
    Wrapper to make GPU engine compatible with benchmark framework
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.engine = GPUYOLOv8Engine(model_path, device)
        self.frame_times = deque(maxlen=1000)
        self.preprocess_times = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        self.postprocess_times = deque(maxlen=1000)
    
    def initialize(self):
        """Initialize the engine"""
        self.engine.initialize()
    
    def infer(self, frame: np.ndarray) -> Tuple[List, Dict[str, float]]:
        """Run inference with timing"""
        detections, timing = self.engine.infer(frame)
        
        # Track timing for benchmark
        self.preprocess_times.append(timing['preprocess_ms'])
        self.inference_times.append(timing['inference_ms'])
        self.postprocess_times.append(timing['postprocess_ms'])
        self.frame_times.append(timing['total_ms'])
        
        return detections, timing
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'avg_fps': 1000.0 / np.mean(self.frame_times) if self.frame_times else 0,
            'avg_preprocess_ms': np.mean(self.preprocess_times) if self.preprocess_times else 0,
            'avg_inference_ms': np.mean(self.inference_times) if self.inference_times else 0,
            'avg_postprocess_ms': np.mean(self.postprocess_times) if self.postprocess_times else 0,
            'avg_total_ms': np.mean(self.frame_times) if self.frame_times else 0,
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.engine.cleanup()

# Utility functions
def download_yolov8_model(model_name: str = "yolov8s.pt") -> str:
    """Download YOLOv8 model for GPU comparison"""
    from ultralytics import YOLO
    
    # This will automatically download the model if not present
    model = YOLO(model_name)
    return model.model_path if hasattr(model, 'model_path') else model_name

def convert_npu_to_gpu_format(npu_detections: List) -> List:
    """Convert NPU detection format to GPU-compatible format"""
    gpu_detections = []
    
    for det in npu_detections:
        if hasattr(det, 'x1'):  # Detection object
            gpu_det = {
                'bbox': [det.x1, det.y1, det.x2, det.y2],
                'confidence': det.confidence,
                'class_id': det.class_id,
                'class_name': det.class_name
            }
        else:  # Array format
            gpu_det = {
                'bbox': det[:4],
                'confidence': det[4],
                'class_id': int(det[5]),
                'class_name': None
            }
        
        gpu_detections.append(gpu_det)
    
    return gpu_detections

# Test function
def test_gpu_engine():
    """Test GPU engine with sample image"""
    import cv2
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # Test with Ultralytics (easiest)
        engine = GPUYOLOv8Engine("yolov8s.pt", engine_type="ultralytics")
        engine.initialize()
        
        detections, timing = engine.infer(test_img)
        
        print(f"GPU Test Results:")
        print(f"  Detections: {len(detections)}")
        print(f"  Timing: {timing}")
        
        engine.cleanup()
        
        return True
        
    except Exception as e:
        print(f"GPU test failed: {e}")
        return False

if __name__ == "__main__":
    test_gpu_engine()