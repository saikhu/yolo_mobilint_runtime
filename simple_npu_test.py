"""
Super simple NPU test script for YOLOv8 model
Tests all core modes with dummy data
"""

import numpy as np
import maccel

# Hardcoded configuration

#MXQ_PATH = "models/yolov8l_global_v4.mxq"  # Your model file

# MXQ_PATH = "/home/mobilint/Desktop/yolov8l/model_quant/yolov8s_best_globalCore_v0.mxq"
# MXQ_PATH = "models/yolov8s_best_globalCore_v1.mxq"
MXQ_PATH = "models/yolov11_NIPA_Data_2025_v8.mxq"

DEVICE_ID = 0                         # NPU device 0
IMG_HEIGHT = 1280
IMG_WIDTH = 1280
IMG_CHANNELS = 3

print("=" * 50)
print("     MOBILINT NPU SIMPLE TEST SCRIPT")
print("=" * 50)
print(f"Model: {MXQ_PATH}")
print(f"Device: {DEVICE_ID}")
print(f"Input: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}")
print("=" * 50)

# Step 1: Create Accelerator
print("\n[STEP 1] Creating Accelerator...")
try:
    acc = maccel.Accelerator(DEVICE_ID)
    print("  ✓ SUCCESS - Accelerator created")
except Exception as e:
    print(f"  ✗ FAILED to create Accelerator!")
    print(f"  Error: {e}")
    exit(1) 

# Create dummy input data (once for all tests)
print("\n[PREPARING] Creating dummy input data...")
dummy_data = np.full((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 0.5, dtype=np.float32)
print(f"  ✓ Dummy data created, shape: {dummy_data.shape}, dtype: {dummy_data.dtype}")

print("\n" + "=" * 50)
print("        TESTING DIFFERENT CORE MODES")
print("=" * 50)

# Test configurations
test_configs = [
    ("DEFAULT (Single, all cores)", lambda: maccel.ModelConfig()),
    ("GLOBAL8 CORE", lambda: create_global8()),
] 


def create_global8():
    config = maccel.ModelConfig()
    config.set_global8_core_mode()
    return config

# Run tests
test_num = 1
for test_name, config_func in test_configs:
    print(f"\n[TEST {test_num}] {test_name}")
    print("-" * 40)
    
    try:
        # Create configuration
        config = config_func()
        
        # Load model
        print("  Loading model...")
        model = maccel.Model(MXQ_PATH, config)
        print("  ✓ Model loaded")
        
        # Launch on NPU
        print("  Launching on NPU...")
        model.launch(acc)
        print("  ✓ Model launched")
        
        # Run inference
        print("  Running inference...")
        result = model.infer([dummy_data])
        print(f"  ✓ SUCCESS! Outputs: {len(result)}")
        
        # Show output details for successful runs
        for i, output in enumerate(result):
            print(f"    Output {i}:")
            print(f"      Shape: {output.shape}")
            print(f"      Elements: {output.size}")
            print(f"      First 5 values: {output.flatten()[:5]}")
            
    except Exception as e:
        print(f"  ✗ FAILED!")
        print(f"  Error: {e}")
        if "global" in test_name.lower():
            print(f"  (This is expected if model wasn't compiled for {test_name.split()[0].lower()} mode)")
    
    test_num += 1

print("\n" + "=" * 50)
print("           TEST COMPLETE")
print("=" * 50)
print("Check which modes worked with your model.")
print("The model must be compiled for the specific")
print("mode you want to use (single/global).")
print("=" * 50)
