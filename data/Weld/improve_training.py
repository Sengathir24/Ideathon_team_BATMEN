# =====================================================
# GPU Configuration & Setup
# =====================================================
import os
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f
from ultralytics.nn.modules.block import CBAM  # Correct CBAM import

# Fix for Windows OpenMP issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
# Dataset Cleaning & Preparation
# =====================================================
def clean_dataset():
    """Clean duplicate labels and invalid annotations"""
    from glob import glob
    
    for split in ['train', 'valid']:
        label_dir = f'D:/data/Weld/{split}/labels'
        for file in glob(f'{label_dir}/*.txt'):
            with open(file, 'r') as f:
                lines = f.readlines()
            # Remove duplicates and invalid entries
            valid_lines = []
            seen = set()
            for line in lines:
                stripped = line.strip()
                if stripped and len(stripped.split()) == 5 and stripped not in seen:
                    valid_lines.append(stripped)
                    seen.add(stripped)
            with open(file, 'w') as f:
                f.write('\n'.join(valid_lines))

clean_dataset()

# =====================================================
# Dataset Configuration
# =====================================================
BASE_DIR = r"D:\data\Weld"
TRAIN_IMAGES = os.path.join(BASE_DIR, "train", "images")
VALID_IMAGES = os.path.join(BASE_DIR, "valid", "images")
TEST_IMAGES = os.path.join(BASE_DIR, "test", "images")

def create_data_yaml():
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

create_data_yaml()

# =====================================================
# Model Architecture Modification (with CBAM Attention)
# =====================================================
def create_attention_model():
    """Add CBAM attention to YOLOv8n backbone"""
    model = YOLO('yolov8n.pt')
    
    # Access model layers
    layers = model.model.model
    
    # Add CBAM after each C2f layer
    for i, layer in enumerate(layers):
        if isinstance(layer, C2f):
            # Get output channels from layer
            out_channels = layer.cv2.conv.out_channels
            
            # Create CBAM module
            cbam = CBAM(channels=out_channels)
            
            # Insert CBAM after C2f layer
            layers[i] = torch.nn.Sequential(
                layer,
                cbam
            ).to('cuda')
    
    return model.to('cuda')

# =====================================================
# Optimized Training Configuration
# =====================================================
def train_model():
    model = create_attention_model()
    
    results = model.train(
        data="custom_data.yaml",
        epochs=300,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        project='welding_defects',
        name='final_training',
        val=True,
        augment=True,
        cache=False,
        amp=True,
        cos_lr=True,
        # Hyperparameters
        lr0=0.0005,
        lrf=0.001,
        momentum=0.96,
        weight_decay=0.0001,
        warmup_epochs=5,
        box=5.0,
        cls=0.25,
        # Augmentation
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=10,
        scale=0.7,
        shear=0.2,
        perspective=0.001,
        mixup=0.2,
        copy_paste=0.3,
        mosaic=0.8,
        close_mosaic=20
    )
    return model

# =====================================================
# Enhanced Inference with TTA
# =====================================================
def detect_with_tta(model, img_path, conf=0.3):
    results = model(
        img_path,
        conf=conf,
        iou=0.5,
        device=0,
        augment=True,
        agnostic_nms=True
    )
    
    defect_classes = {0, 1, 2, 4, 5}
    detected = {int(box.cls.item()) for box in results[0].boxes}
    results[0].show()
    
    print(f"\nDetected: {[model.names[c] for c in detected]}")
    print("Defect Found" if detected & defect_classes else "No Defect")

# =====================================================
# Main Execution
# =====================================================
if __name__ == "__main__":
    # Verify dataset paths
    for path in [TRAIN_IMAGES, VALID_IMAGES, TEST_IMAGES]:
        assert os.path.exists(path), f"Path {path} does not exist"
    
    # Train model with attention
    model = train_model()
    
    # Validate
    metrics = model.val()
    print(f"\nmAP50-95: {metrics.box.map:.2f}")
    print(f"mAP50: {metrics.box.map50:.2f}")
    
    # Test detection
    test_image = os.path.join(TEST_IMAGES, "sample.jpg")
    detect_with_tta(model, test_image)