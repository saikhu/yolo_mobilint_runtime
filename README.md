# NPU Inference Module for Backend Integration

## Files Included

### 📁 Project Structure
```
npu_inference_module/
├── npu_inference.py    # Core NPU inference module
├── main.py             # Usage examples
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## 📋 requirements.txt
```
numpy>=1.19.0
opencv-python>=4.5.0
torch>=1.9.0  # For postprocessing (can be optimized out if needed)
maccel  # Mobilint NPU runtime library
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Note: maccel library must be installed according to Mobilint documentation
```

### 2. Basic Usage

```python
from npu_inference import NPUInferenceEngine, InferenceConfig
import cv2

# Configure the engine
config = InferenceConfig(
    model_path="path/to/your/model.mxq",
    conf_threshold=0.5,
    iou_threshold=0.5
)

# Initialize engine
engine = NPUInferenceEngine(config)
engine.initialize()

# Load and process image
image = cv2.imread("test.jpg")
detections = engine.detect(image)

# Process results
for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f} at [{det.x1}, {det.y1}, {det.x2}, {det.y2}]")

# Clean up
engine.cleanup()
```

### 3. Using Context Manager (Recommended)

```python
with NPUInferenceEngine(config) as engine:
    image = cv2.imread("test.jpg")
    detections = engine.detect(image)
    # Engine automatically cleaned up when exiting context
```

---

## 🔧 API Reference

### InferenceConfig

Configuration dataclass for the inference engine.

**Parameters:**
- `model_path` (str): Path to the .mxq model file
- `img_size` (tuple): Input image size, default (640, 640)
- `num_classes` (int): Number of detection classes, default 17
- `conf_threshold` (float): Confidence threshold, default 0.5
- `iou_threshold` (float): NMS IOU threshold, default 0.5
- `use_global8_core` (bool): Use all 8 NPU cores, default True
- `device_id` (int): NPU device ID, default 0

### NPUInferenceEngine

Main inference engine class.

**Methods:**
- `initialize()`: Initialize NPU hardware and load model
- `detect(image)`: Run detection on single image
- `detect_batch(images)`: Process multiple images
- `preprocess(image)`: Preprocess image for inference
- `infer(preprocessed_image)`: Run NPU inference
- `postprocess(outputs, metadata)`: Convert raw outputs to detections
- `cleanup()`: Release NPU resources
- `get_stats()`: Get engine statistics

### Detection

Detection result dataclass.

**Attributes:**
- `x1`, `y1`, `x2`, `y2` (float): Bounding box coordinates
- `confidence` (float): Detection confidence score
- `class_id` (int): Class index
- `class_name` (str): Class name (optional)

---

## 💻 Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2

app = FastAPI()
engine = None

@app.on_event("startup")
async def startup():
    global engine
    config = InferenceConfig(model_path="model.mxq")
    engine = NPUInferenceEngine(config)
    engine.initialize()

@app.on_event("shutdown")
async def shutdown():
    global engine
    if engine:
        engine.cleanup()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run detection
    detections = engine.detect(image)
    
    # Format response
    results = {
        "detections": [
            {
                "class": det.class_name,
                "confidence": float(det.confidence),
                "bbox": [det.x1, det.y1, det.x2, det.y2]
            }
            for det in detections
        ]
    }
    
    return JSONResponse(content=results)
```

### Flask Integration

```python
from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

# Initialize engine at startup
config = InferenceConfig(model_path="model.mxq")
engine = NPUInferenceEngine(config)
engine.initialize()

@app.route('/detect', methods=['POST'])
def detect():
    # Get image from request
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Run detection
    detections = engine.detect(image)
    
    # Return results
    return jsonify({
        'detections': [
            {
                'class': det.class_name,
                'confidence': det.confidence,
                'bbox': [det.x1, det.y1, det.x2, det.y2]
            }
            for det in detections
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## ⚡ Performance Optimization Tips

1. **Initialize Once**: Initialize the engine once at application startup, not for each request
2. **Use Context Manager**: For batch processing, use the context manager for automatic cleanup
3. **Thread Safety**: The engine is NOT thread-safe. Use one engine per thread or implement locking
4. **Batch Processing**: Process multiple images together when possible
5. **Core Configuration**: Use `use_global8_core=True` for maximum performance

---

## 🔍 Troubleshooting

### Common Issues

1. **NPU Not Found**
   - Check if NPU driver is installed: `ls /dev/aries*`
   - Verify maccel library installation

2. **Model Loading Error**
   - Ensure .mxq file path is correct
   - Check model compatibility with NPU version

3. **Out of Memory**
   - Reduce batch size
   - Check if other processes are using NPU

4. **Low Performance**
   - Enable global8_core mode
   - Check if NPU is thermal throttling
   - Ensure no other heavy processes on NPU

---

## 📊 Monitoring

Get engine statistics:

```python
stats = engine.get_stats()
print(f"Initialized: {stats['initialized']}")
print(f"Device ID: {stats['device_id']}")
print(f"Model: {stats['model_path']}")
```

---

## 📧 Support

For NPU-specific issues, refer to Mobilint documentation or contact their support.
For integration questions, contact the AI team.

---

## 🔄 Version History

- v1.0.0: Initial production release
  - Core NPU inference functionality
  - Support for YOLOv8 models
  - Context manager support
  - Batch processing capability