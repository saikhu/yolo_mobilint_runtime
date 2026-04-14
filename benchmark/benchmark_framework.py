#!/usr/bin/env python3
"""
Fixed NPU vs GPU YOLOv8 Comparison Framework
Corrected implementation with proper imports and error handling
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

try:
    from yolo_mobilint_runtime.src.npu_inference_yolov8 import NPUInferenceEngine, InferenceConfig
except ImportError as e:
    logging.error(f"Cannot import npu_inference: {e}")
    NPUInferenceEngine = None

try:
    from gpu_yolov8_inference import GPUYOLOv8Engine
except ImportError as e:
    logging.error(f"Cannot import gpu_yolov8_inference: {e}")
    GPUYOLOv8Engine = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedSimpleComparison:
    """Fixed NPU vs GPU comparison with proper error handling"""
    
    def __init__(self, npu_model_path, gpu_model_path=None):
        self.npu_model_path = npu_model_path
        self.gpu_model_path = gpu_model_path or "yolov8s.pt"
        self.results = {
            'npu': {'fps': [], 'latency': [], 'detections': []},
            'gpu': {'fps': [], 'latency': [], 'detections': []}
        }
        
        # Check if files exist
        if not os.path.exists(self.npu_model_path):
            logger.error(f"NPU model not found: {self.npu_model_path}")
    
    def test_single_image(self, image_path=None):
        """Test both engines on a single image"""
        logger.info("Running single image test...")
        
        # Create or load test image
        if image_path and os.path.exists(image_path):
            test_img = cv2.imread(image_path)
        else:
            # Create synthetic test image
            test_img = self._create_test_image()
            logger.info("Using synthetic test image")
        
        # Test NPU
        npu_results = self._test_npu_single(test_img)
        
        # Test GPU
        gpu_results = self._test_gpu_single(test_img)
        
        # Compare results
        self._print_single_comparison(npu_results, gpu_results)
        
        return npu_results, gpu_results
    
    def test_video_stream(self, stream_path, duration=60):
        """Test both engines on video stream"""
        logger.info(f"Testing video stream for {duration} seconds...")
        
        if not os.path.exists(stream_path):
            logger.error(f"Video file not found: {stream_path}")
            return None, None
        
        # Test NPU
        logger.info("Testing NPU...")
        npu_stats = self._test_engine_on_stream(
            self._create_npu_engine, stream_path, duration, "NPU"
        )
        
        # Test GPU
        logger.info("Testing GPU...")
        gpu_stats = self._test_engine_on_stream(
            self._create_gpu_engine, stream_path, duration, "GPU"
        )
        
        # Compare results
        self._print_stream_comparison(npu_stats, gpu_stats)
        
        return npu_stats, gpu_stats
    
    def test_multi_stream(self, streams, duration=60):
        """Test both engines with multiple concurrent streams"""
        logger.info(f"Testing {len(streams)} concurrent streams...")
        
        results = {}
        
        for num_streams in [1, 2, 4]:
            if num_streams > len(streams):
                continue
                
            logger.info(f"Testing with {num_streams} stream(s)")
            
            # Test NPU
            npu_stats = self._test_concurrent_streams(
                'NPU', streams[:num_streams], duration
            )
            
            # Test GPU
            gpu_stats = self._test_concurrent_streams(
                'GPU', streams[:num_streams], duration
            )
            
            results[num_streams] = {
                'npu': npu_stats,
                'gpu': gpu_stats
            }
        
        self._print_multi_stream_results(results)
        return results
    
    def _create_test_image(self):
        """Create synthetic test image with objects"""
        img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add some rectangular shapes to simulate objects
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.rectangle(img, (300, 200), (450, 350), (255, 0, 0), -1)
        cv2.rectangle(img, (500, 50), (600, 150), (0, 0, 255), -1)
        
        return img
    
    def _create_npu_engine(self):
        """Create NPU engine with proper error handling"""
        if NPUInferenceEngine is None:
            raise ImportError("NPU inference engine not available")
        
        config = InferenceConfig(
            model_path=self.npu_model_path,
            conf_threshold=0.5,
            use_global8_core=True
        )
        engine = NPUInferenceEngine(config)
        engine.initialize()
        return engine
    
    def _create_gpu_engine(self):
        """Create GPU engine with proper error handling"""
        if GPUYOLOv8Engine is None:
            raise ImportError("GPU inference engine not available")
        
        # Try Ultralytics first
        try:
            from ultralytics import YOLO
            engine = GPUYOLOv8Engine(self.gpu_model_path, engine_type="ultralytics")
        except ImportError:
            logger.warning("Ultralytics not available, trying ONNX")
            if self.gpu_model_path.endswith('.pt'):
                # Convert to ONNX path or use default
                self.gpu_model_path = self.gpu_model_path.replace('.pt', '.onnx')
            engine = GPUYOLOv8Engine(self.gpu_model_path, engine_type="onnx")
        
        engine.initialize()
        return engine
    
    def _test_npu_single(self, image):
        """Test NPU on single image"""
        try:
            engine = self._create_npu_engine()
            
            # Warm up
            for _ in range(3):
                _ = engine.detect(image)
            
            # Actual test
            start_time = time.perf_counter()
            detections = engine.detect(image)
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000
            fps = 1000 / latency if latency > 0 else 0
            
            engine.cleanup()
            
            return {
                'device': 'NPU',
                'detections': len(detections),
                'latency_ms': latency,
                'fps': fps,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"NPU test failed: {e}")
            return {'device': 'NPU', 'success': False, 'error': str(e)}
    
    def _test_gpu_single(self, image):
        """Test GPU on single image"""
        try:
            engine = self._create_gpu_engine()
            
            # Warm up
            for _ in range(3):
                _ = engine.detect(image)
            
            # Actual test
            start_time = time.perf_counter()
            detections = engine.detect(image)
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000
            fps = 1000 / latency if latency > 0 else 0
            
            engine.cleanup()
            
            return {
                'device': 'GPU',
                'detections': len(detections),
                'latency_ms': latency,
                'fps': fps,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"GPU test failed: {e}")
            return {'device': 'GPU', 'success': False, 'error': str(e)}
    
    def _test_engine_on_stream(self, engine_factory, stream_path, duration, device_name):
        """Test engine on video stream"""
        cap = cv2.VideoCapture(stream_path)
        if not cap.isOpened():
            logger.error(f"Failed to open stream: {stream_path}")
            return None
        
        frame_times = []
        detection_counts = []
        
        try:
            engine = engine_factory()
        except Exception as e:
            logger.error(f"Failed to create {device_name} engine: {e}")
            cap.release()
            return None
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    # Restart video if it ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                # Skip every other frame to reduce load
                if frame_count % 2 == 0:
                    frame_start = time.perf_counter()
                    
                    try:
                        detections = engine.detect(frame)
                        frame_end = time.perf_counter()
                        
                        frame_time = (frame_end - frame_start) * 1000
                        frame_times.append(frame_time)
                        detection_counts.append(len(detections))
                    except Exception as e:
                        logger.warning(f"{device_name} inference failed: {e}")
                
                frame_count += 1
            
        finally:
            cap.release()
            try:
                engine.cleanup()
            except:
                pass
        
        if frame_times:
            return {
                'avg_fps': 1000 / np.mean(frame_times),
                'avg_latency_ms': np.mean(frame_times),
                'total_frames': len(frame_times),
                'avg_detections': np.mean(detection_counts),
                'frame_times': frame_times[-100:]  # Keep last 100 for analysis
            }
        else:
            return None
    
    def _test_concurrent_streams(self, device_type, streams, duration):
        """Test concurrent streams (simplified implementation)"""
        results = []
        
        for i, stream in enumerate(streams):
            if not os.path.exists(stream):
                logger.warning(f"Stream file not found: {stream}")
                continue
                
            logger.info(f"  Testing stream {i+1}/{len(streams)}")
            
            if device_type == 'NPU':
                engine_factory = self._create_npu_engine
            else:
                engine_factory = self._create_gpu_engine
            
            # Test for shorter duration per stream
            stream_duration = max(10, duration // len(streams))
            stats = self._test_engine_on_stream(engine_factory, stream, stream_duration, device_type)
            
            if stats:
                results.append(stats)
        
        # Aggregate results
        if results:
            return {
                'total_streams': len(streams),
                'avg_fps_per_stream': np.mean([r['avg_fps'] for r in results]),
                'total_throughput_fps': sum([r['avg_fps'] for r in results]),
                'avg_latency_ms': np.mean([r['avg_latency_ms'] for r in results]),
                'success_rate': len(results) / len(streams)
            }
        else:
            return None
    
    def _print_single_comparison(self, npu_results, gpu_results):
        """Print single image comparison results"""
        print("\n" + "="*60)
        print("           SINGLE IMAGE COMPARISON")
        print("="*60)
        
        if npu_results.get('success') and gpu_results.get('success'):
            print(f"{'Metric':<20} {'NPU':<15} {'GPU':<15} {'Winner'}")
            print("-" * 60)
            
            # FPS comparison
            npu_fps = npu_results['fps']
            gpu_fps = gpu_results['fps']
            fps_winner = 'NPU' if npu_fps > gpu_fps else 'GPU'
            print(f"{'FPS':<20} {npu_fps:<15.1f} {gpu_fps:<15.1f} {fps_winner}")
            
            # Latency comparison
            npu_lat = npu_results['latency_ms']
            gpu_lat = gpu_results['latency_ms']
            lat_winner = 'NPU' if npu_lat < gpu_lat else 'GPU'
            print(f"{'Latency (ms)':<20} {npu_lat:<15.1f} {gpu_lat:<15.1f} {lat_winner}")
            
            # Detection count
            npu_det = npu_results['detections']
            gpu_det = gpu_results['detections']
            print(f"{'Detections':<20} {npu_det:<15} {gpu_det:<15}")
            
        else:
            print("Test failed for one or both devices:")
            if not npu_results.get('success'):
                print(f"  NPU Error: {npu_results.get('error', 'Unknown')}")
            if not gpu_results.get('success'):
                print(f"  GPU Error: {gpu_results.get('error', 'Unknown')}")
        
        print("="*60)
    
    def _print_stream_comparison(self, npu_stats, gpu_stats):
        """Print video stream comparison results"""
        print("\n" + "="*60)
        print("           VIDEO STREAM COMPARISON")
        print("="*60)
        
        if npu_stats and gpu_stats:
            print(f"{'Metric':<25} {'NPU':<15} {'GPU':<15}")
            print("-" * 60)
            print(f"{'Avg FPS':<25} {npu_stats['avg_fps']:<15.1f} {gpu_stats['avg_fps']:<15.1f}")
            print(f"{'Avg Latency (ms)':<25} {npu_stats['avg_latency_ms']:<15.1f} {gpu_stats['avg_latency_ms']:<15.1f}")
            print(f"{'Total Frames':<25} {npu_stats['total_frames']:<15} {gpu_stats['total_frames']:<15}")
            print(f"{'Avg Detections':<25} {npu_stats['avg_detections']:<15.1f} {gpu_stats['avg_detections']:<15.1f}")
        else:
            print("Stream test failed")
        
        print("="*60)
    
    def _print_multi_stream_results(self, results):
        """Print multi-stream comparison results"""
        print("\n" + "="*80)
        print("                 MULTI-STREAM COMPARISON")
        print("="*80)
        
        print(f"{'Streams':<10} {'NPU FPS':<15} {'GPU FPS':<15} {'NPU Latency':<15} {'GPU Latency':<15}")
        print("-" * 80)
        
        for num_streams, data in results.items():
            npu_data = data['npu']
            gpu_data = data['gpu']
            
            if npu_data and gpu_data:
                print(f"{num_streams:<10} "
                      f"{npu_data['total_throughput_fps']:<15.1f} "
                      f"{gpu_data['total_throughput_fps']:<15.1f} "
                      f"{npu_data['avg_latency_ms']:<15.1f} "
                      f"{gpu_data['avg_latency_ms']:<15.1f}")
        
        print("="*80)

def main():
    """Main execution function"""
    
    # Configuration
    NPU_MODEL = "../models/yolov8s_best_globalCore_v1.mxq"
    GPU_MODEL = "../models/yolov8s_best.pt"  # Will be downloaded automatically
    
    # Test streams
    TEST_STREAMS = [
        "input/0.mp4",
        "input/1.mp4",
    ]
    
    print("NPU vs GPU YOLOv8 Comparison Tool")
    print("=" * 50)
    print(f"NPU Model: {NPU_MODEL}")
    print(f"GPU Model: {GPU_MODEL}")
    print("=" * 50)
    
    # Check if NPU model exists
    if not os.path.exists(NPU_MODEL):
        print(f"Error: NPU model not found at {NPU_MODEL}")
        print("Please update the model path")
        return
    
    # Create comparison instance
    comparison = FixedSimpleComparison(NPU_MODEL, GPU_MODEL)
    
    # Run tests
    try:
        print("\n[TEST 1] Single Image Comparison")
        comparison.test_single_image()
        
        # Find existing video file
        test_video = None
        for video in TEST_STREAMS:
            if os.path.exists(video):
                test_video = video
                break
        
        if test_video:
            print("\n[TEST 2] Video Stream Comparison")
            comparison.test_video_stream(test_video, duration=30)
            
            print("\n[TEST 3] Multi-Stream Comparison")
            comparison.test_multi_stream([test_video], duration=60)
        else:
            print("\nSkipping video tests - no video files found")
            print(f"Expected files: {TEST_STREAMS}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        logger.exception("Test execution failed")

if __name__ == "__main__":
    main()