# Step 1: Import necessary modules
from ultralytics import YOLO
import torch
import os

# =====================================================
# Step 2: Configure dataset paths (MODIFY THESE VARIABLES)
# =====================================================
BASE_DIR = r"D:\ML2\Weld quality inspection - Segmentation"  # <--- UPDATE THIS TO YOUR DATASET PATH
TRAIN_IMAGES = os.path.join(BASE_DIR, "train", "images")
TRAIN_LABELS = os.path.join(BASE_DIR, "train", "labels")
VALID_IMAGES = os.path.join(BASE_DIR, "valid", "images")
VALID_LABELS = os.path.join(BASE_DIR, "valid", "labels")
TEST_IMAGES = os.path.join(BASE_DIR, "test", "images")
TEST_LABELS = os.path.join(BASE_DIR, "test", "labels")

# Create data.yaml configuration file programmatically
def create_data_yaml():
    data_config = f"""
train: {TRAIN_IMAGES}
val: {VALID_IMAGES}
test: {TEST_IMAGES}

nc: 6
names: ['Bad Welding', 'Crack', 'Excess Reinforcement', 'Good Welding', 'Porosity', 'Spatters']
"""
    with open("custom_data.yaml", "w") as f:
        f.write(data_config.strip())
    print(f"Created custom_data.yaml with:\n{data_config}")

# Initialize configuration
create_data_yaml()

# =====================================================
# Step 3: Optimized Training Configuration
# =====================================================
def train_model():
    # Ensure GPU is used
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    model = YOLO('yolov8n.pt')  # Start with pretrained weights
    
    # Optimize for RTX 3050 (6GB VRAM)
    results = model.train(
        data="custom_data.yaml",         # Path to data.yaml
        epochs=200,                     # Increase epochs for better convergence
        imgsz=640,                      # Image size (balance between resolution and memory)
        batch=8,                        # Batch size optimized for 6GB VRAM
        device=device,                  # Explicitly set device (0 for GPU)
        workers=4,                      # Number of CPU threads for data loading
        project='welding_defects',
        name='gpu_training_optimized',
        val=True,                       # Enable validation during training
        augment=True,                   # Enable data augmentation
        cache='disk',                   # Cache images on disk for deterministic training
        amp=True,                       # Mixed precision training (float16 + float32)
        lr0=0.001,                      # Initial learning rate
        lrf=0.01,                       # Final learning rate (as a fraction of lr0)
        cos_lr=True,                    # Use cosine learning rate decay
        optimizer='AdamW',              # AdamW optimizer for better generalization
        momentum=0.937,                 # Momentum for SGD (if using SGD)
        weight_decay=0.0005,            # Regularization to prevent overfitting
        close_mosaic=10,                # Disable mosaic augmentation in the last 10 epochs
        mosaic=1.0,                     # Enable mosaic augmentation
        mixup=0.1,                      # Mixup probability
        copy_paste=0.1,                 # Copy-paste augmentation for overlapping objects
        hsv_h=0.015,                    # HSV hue augmentation
        hsv_s=0.7,                      # HSV saturation augmentation
        hsv_v=0.4,                      # HSV value augmentation
        degrees=10,                     # Random rotation
        translate=0.1,                  # Random translation
        scale=0.5,                      # Random scaling
        shear=0.0,                      # Random shear
        perspective=0.0,                # Random perspective transformation
        flipud=0.0,                     # Vertical flip probability
        fliplr=0.5                      # Horizontal flip probability
    )
    return model

# =====================================================
# Step 4: Evaluation and Inference
# =====================================================
def evaluate_model(model):
    metrics = model.val()
    print(f"\nValidation Results:")
    print(f"mAP50-95: {metrics.box.map:.2f}")
    print(f"mAP50: {metrics.box.map50:.2f}")
    print(f"Precision: {metrics.box.mp:.2f}")
    print(f"Recall: {metrics.box.mr:.2f}")

def detect_and_classify(model, img_path, conf=0.5):
    # Ensure GPU is used
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    # Check if the image path exists
    assert os.path.exists(img_path), f"Image path not found: {img_path}"
    
    results = model(img_path, conf=conf, device=device)
    
    # Class indices for defects (0-5, excluding 3: Good Welding)
    defect_classes = {0, 1, 2, 4, 5}
    detected_classes = set()
    
    for box in results[0].boxes:
        detected_classes.add(int(box.cls.item()))
    
    has_defect = not detected_classes.isdisjoint(defect_classes)
    
    # Show results
    results[0].show()
    print("\nDetection Results:")
    print(f"Detected classes: {[model.names[c] for c in detected_classes]}")
    print("Defect Detected" if has_defect else "No Defect Found")

# =====================================================
# Main Execution
# =====================================================
if _name_ == "_main_":
    # Print paths for debugging
    print(f"Train Images Path: {TRAIN_IMAGES}")
    print(f"Validation Images Path: {VALID_IMAGES}")
    print(f"Test Images Path: {TEST_IMAGES}")

    # Verify paths
    assert os.path.exists(TRAIN_IMAGES), f"Training images path not found: {TRAIN_IMAGES}"
    assert os.path.exists(VALID_IMAGES), f"Validation images path not found: {VALID_IMAGES}"
    assert os.path.exists(TEST_IMAGES), f"Test images path not found: {TEST_IMAGES}"
    
    # Train model (uncomment to train)
    model = train_model()  # <--- UNCOMMENT THIS LINE TO START TRAINING
    
    # Load trained model
    # model = YOLO('welding_defects/gpu_training/weights/best.pt')  # <--- LOAD TRAINED WEIGHTS IF NEEDED
    
    # Evaluate
    evaluate_model(model)
    
    # Test detection
    test_image = os.path.join(TEST_IMAGES, "sample.jpg")  # Update with your test image
    print(f"Test Image Path: {test_image}")  # Debug statement
    assert os.path.exists(test_image), f"Test image path not found: {test_image}"  # Verify test image path
    detect_and_classify(model, test_image, conf=0.6)
