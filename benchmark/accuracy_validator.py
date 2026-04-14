"""
Accuracy Validation Script
Compare NPU vs GPU detection accuracy using ground truth or cross-validation
"""

import cv2
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Single detection result"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: Optional[str] = None

@dataclass
class AccuracyResults:
    """Accuracy comparison results"""
    total_frames: int
    exact_matches: int
    close_matches: int  # IoU > threshold
    npu_unique: int
    gpu_unique: int
    avg_iou: float
    avg_conf_diff: float
    per_class_accuracy: Dict[str, float]

class AccuracyValidator:
    """Compare detection accuracy between NPU and GPU"""
    
    def __init__(self, iou_threshold: float = 0.5, conf_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.class_names = [
            "backhoe_loader", "cement_truck", "compactor", "dozer", "dump_truck",
            "excavator", "grader", "mobile_crane", "tower_crane", "wheel_loader",
            "worker", "Hardhat", "Red_Hardhat", "scaffolds", "Lifted Load",
            "Crane_Hook", "Hook"
        ]
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
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
    
    def match_detections(self, npu_dets: List[Detection], gpu_dets: List[Detection]) -> Dict:
        """Match detections between NPU and GPU"""
        matches = []
        npu_matched = set()
        gpu_matched = set()
        
        # Find matches
        for i, npu_det in enumerate(npu_dets):
            best_match = None
            best_iou = 0.0
            
            for j, gpu_det in enumerate(gpu_dets):
                if j in gpu_matched or npu_det.class_id != gpu_det.class_id:
                    continue
                
                iou = self.calculate_iou(
                    [npu_det.x1, npu_det.y1, npu_det.x2, npu_det.y2],
                    [gpu_det.x1, gpu_det.y1, gpu_det.x2, gpu_det.y2]
                )
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_match = j
                    best_iou = iou
            
            if best_match is not None:
                matches.append({
                    'npu_idx': i,
                    'gpu_idx': best_match,
                    'iou': best_iou,
                    'conf_diff': abs(npu_det.confidence - gpu_dets[best_match].confidence),
                    'class_id': npu_det.class_id
                })
                npu_matched.add(i)
                gpu_matched.add(best_match)
        
        # Count unmatched
        npu_unique = len(npu_dets) - len(npu_matched)
        gpu_unique = len(gpu_dets) - len(gpu_matched)
        
        return {
            'matches': matches,
            'npu_unique': npu_unique,
            'gpu_unique': gpu_unique,
            'npu_matched': npu_matched,
            'gpu_matched': gpu_matched
        }
    
    def validate_accuracy(self, npu_engine, gpu_engine, test_frames: List[np.ndarray]) -> AccuracyResults:
        """Validate accuracy across test frames"""
        total_matches = 0
        total_close_matches = 0
        total_npu_unique = 0
        total_gpu_unique = 0
        all_ious = []
        all_conf_diffs = []
        class_stats = {name: {'matches': 0, 'total': 0} for name in self.class_names}
        
        logger.info(f"Validating accuracy on {len(test_frames)} frames")
        
        for frame_idx, frame in enumerate(test_frames):
            # Get detections from both engines
            npu_detections, _ = npu_engine.infer(frame)
            gpu_detections, _ = gpu_engine.infer(frame)
            
            # Convert to Detection objects (assuming engines return compatible format)
            npu_dets = self._convert_to_detection_objects(npu_detections)
            gpu_dets = self._convert_to_detection_objects(gpu_detections)
            
            # Match detections
            match_result = self.match_detections(npu_dets, gpu_dets)
            
            # Update statistics
            matches = match_result['matches']
            total_matches += len([m for m in matches if m['iou'] > 0.9])  # Exact matches
            total_close_matches += len(matches)  # Close matches
            total_npu_unique += match_result['npu_unique']
            total_gpu_unique += match_result['gpu_unique']
            
            # Collect IoU and confidence differences
            for match in matches:
                all_ious.append(match['iou'])
                all_conf_diffs.append(match['conf_diff'])
                
                # Per-class statistics
                class_name = self.class_names[match['class_id']]
                class_stats[class_name]['matches'] += 1
            
            # Count total detections per class
            for det in npu_dets + gpu_dets:
                if det.class_id < len(self.class_names):
                    class_name = self.class_names[det.class_id]
                    class_stats[class_name]['total'] += 1
            
            if (frame_idx + 1) % 50 == 0:
                logger.info(f"Processed {frame_idx + 1}/{len(test_frames)} frames")
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for class_name, stats in class_stats.items():
            if stats['total'] > 0:
                per_class_accuracy[class_name] = stats['matches'] / stats['total']
            else:
                per_class_accuracy[class_name] = 0.0
        
        return AccuracyResults(
            total_frames=len(test_frames),
            exact_matches=total_matches,
            close_matches=total_close_matches,
            npu_unique=total_npu_unique,
            gpu_unique=total_gpu_unique,
            avg_iou=np.mean(all_ious) if all_ious else 0.0,
            avg_conf_diff=np.mean(all_conf_diffs) if all_conf_diffs else 0.0,
            per_class_accuracy=per_class_accuracy
        )
    
    def _convert_to_detection_objects(self, detections: List) -> List[Detection]:
        """Convert engine output to Detection objects"""
        result = []
        for det in detections:
            if isinstance(det, dict):
                # Dictionary format
                bbox = det.get('bbox', [0, 0, 0, 0])
                result.append(Detection(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                    confidence=det.get('confidence', 0.0),
                    class_id=det.get('class_id', 0),
                    class_name=det.get('class_name')
                ))
            elif len(det) >= 6:
                # Array format [x1, y1, x2, y2, conf, class_id]
                result.append(Detection(
                    x1=det[0], y1=det[1], x2=det[2], y2=det[3],
                    confidence=det[4], class_id=int(det[5]),
                    class_name=self.class_names[int(det[5])] if int(det[5]) < len(self.class_names) else None
                ))
        return result
    
    def save_validation_results(self, results: AccuracyResults, output_path: str):
        """Save validation results to file"""
        with open(output_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        logger.info(f"Validation results saved to {output_path}")
    
    def print_accuracy_summary(self, results: AccuracyResults):
        """Print accuracy validation summary"""
        print("\n" + "="*60)
        print("               ACCURACY VALIDATION RESULTS")
        print("="*60)
        print(f"Total frames processed: {results.total_frames}")
        print(f"Exact matches (IoU > 0.9): {results.exact_matches}")
        print(f"Close matches (IoU > {self.iou_threshold}): {results.close_matches}")
        print(f"NPU unique detections: {results.npu_unique}")
        print(f"GPU unique detections: {results.gpu_unique}")
        print(f"Average IoU: {results.avg_iou:.3f}")
        print(f"Average confidence difference: {results.avg_conf_diff:.3f}")
        
        print("\nPER-CLASS ACCURACY:")
        print("-" * 30)
        for class_name, accuracy in results.per_class_accuracy.items():
            if accuracy > 0:
                print(f"{class_name:<20}: {accuracy:.3f}")
        print("="*60)