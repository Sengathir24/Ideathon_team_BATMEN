# =====================================================
# GPU Configuration & Setup
# =====================================================
import os
import torch

# Fix for Windows OpenMP issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch._C._jit_set_profiling_mode(False)  # Critical fix

# CUDA compatibility checks
print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

# Optimize GPU memory usage
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =====================================================
# Dataset Configuration (UPDATE THESE PATHS)
# =====================================================
BASE_DIR = r"D:\data\Weld"  # <-- VERIFY THIS PATH EXISTS
TRAIN_IMAGES = os.path.join(BASE_DIR, "train", "images")
VALID_IMAGES = os.path.join(BASE_DIR, "valid", "images")
TEST_IMAGES = os.path.join(BASE_DIR, "test", "images")

def create_data_yaml():
    """Generate YOLO-compatible data configuration"""
    train_path = TRAIN_IMAGES.replace("\\", "/")
    val_path = VALID_IMAGES.replace("\\", "/")
    test_path = TEST_IMAGES.replace("\\", "/")
    
    data_config = f"""train: {train_path}
val: {val_path}
test: {test_path}
nc: 6
names: ['Bad Welding', 'Crack', 'Excess Reinforcement', 'Good Welding', 'Porosity', 'Spatters']"""
    
    with open("custom_data.yaml", "w") as f:
        f.write(data_config)
    print(f"\nCreated custom_data.yaml with:\n{data_config}")

# =====================================================
# Training Configuration (Optimized for RTX 3050)
# =====================================================
from ultralytics import YOLO

def train_model():
    """Train YOLOv8 model with hardware-specific optimizations"""
    model = YOLO('yolov8n.pt').to('cuda')
    
    results = model.train(
        data="custom_data.yaml",
        epochs=200,
        imgsz=416,
        batch=8,
        device=0,
        workers=2,
        project='welding_defects',
        name='win_training',
        val=True,
        augment=True,
        cache=False,
        amp=True,
        verbose=True,
        patience=50,
        close_mosaic=0
    )
    return model

# =====================================================
# Main Execution
# =====================================================
if __name__ == "__main__":
    # Verify dataset structure
    print("\n=== Verifying Dataset ===")
    for path in [TRAIN_IMAGES, VALID_IMAGES, TEST_IMAGES]:
        assert os.path.exists(path), f"MISSING: {path}"
        print(f"Verified: {path} contains {len(os.listdir(path))} files")
    
    # Create configuration
    create_data_yaml()
    
    # Train model
    print("\n=== Starting Training ===")
    model = train_model()
    
    # Load best weights
    best_model = YOLO('welding_defects/win_training/weights/best.pt').to('cuda')
    
    # Validate
    print("\n=== Validation Results ===")
    metrics = best_model.val()
    print(f"mAP50-95: {metrics.box.map:.2f}")
    print(f"mAP50: {metrics.box.map50:.2f}")