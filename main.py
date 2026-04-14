"""
Example usage of NPU Inference Module for backend integration.
This demonstrates how to use the NPU inference engine in production.
"""

import os
import cv2
import numpy as np
import logging
from typing import List

# from src.npu_inference import NPUInferenceEngine, InferenceConfig, Detection  # YOLOv8
from src.npu_inference_yolov11 import NPUInferenceEngine, InferenceConfig, Detection  # YOLOv11

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def can_show() -> bool:
    """True if an X/GUI display is available."""
    return bool(os.environ.get("DISPLAY"))


def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw detection boxes on image."""
    img_copy = image.copy()

    for det in detections:
        cv2.rectangle(
            img_copy,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            (0, 255, 0),
            2
        )

        label = f"{det.class_name or det.class_id}: {det.confidence:.2f}"
        cv2.putText(
            img_copy,
            label,
            (int(det.x1), int(det.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return img_copy


def build_output_path(source: str) -> str:
    """
    Build output path as: same folder + same name + _NPU.mp4
    If not a local file path (e.g., RTSP), save in cwd.
    """
    if source and os.path.exists(source):
        base, _ = os.path.splitext(source)
        return base + "_NPU.mp4"
    return os.path.join(os.getcwd(), "stream_NPU.mp4")


def example_video_stream():
    """Example: Process video stream/file."""

    config = InferenceConfig(
        model_path="models/yolov11_NIPA_Data_2025_v8.mxq",
        img_size=(1280, 1280),
        num_classes=14,
        conf_threshold=0.55,
        iou_threshold=0.55,
        use_global8_core=True,
        device_id=0
    )

    # ---- Input source ----
    source = "/home/mobilint/Desktop/usman/test_videos/Hamyang/Instrusion_2/20251023_152922.mp4"
    # source = "rtsp://admin:pass@ip:554/..."  # if needed

    show = can_show()

    with NPUInferenceEngine(config) as engine:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        # Read video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        out_path = None

        if not show:
            out_path = build_output_path(source)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            logger.info(f"No DISPLAY detected. Saving annotated video to: {out_path}")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = engine.detect(frame)

            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Frame {frame_count}: {len(detections)} detections")

            result = draw_detections(frame, detections)

            if show:
                cv2.imshow("Detections", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                writer.write(result)

        cap.release()

        if writer is not None:
            writer.release()
            print(f"\n✅ Saved annotated output: {out_path}")

        if show:
            cv2.destroyAllWindows()


def main():
    print("NPU Inference Engine Examples")
    print("=" * 40)
    print("Video Stream Processing")
    print()

    try:
        example_video_stream()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
