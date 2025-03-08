# Real-Time Defect Detection for Manufacturing (Laptop Version)
# Uses your trained YOLOv8 model and laptop webcam

import cv2
import torch
from ultralytics import YOLO

# =====================================================
# Configuration
# =====================================================
MODEL_PATH = r"D:\data\welding_defects\win_training\weights\best.pt"
CLASS_NAMES = ['Bad Welding', 'Crack', 'Excess Reinforcement', 
               'Good Welding', 'Porosity', 'Spatters']
DEFECT_CLASSES = {0, 1, 2, 4, 5}  # Exclude class 3 (Good Welding)

# Check GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Load model
model = YOLO(MODEL_PATH).to(DEVICE)

# =====================================================
# Real-Time Detection
# =====================================================
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Use default webcam
    confidence = 0.6  # Initial confidence threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame, conf=confidence, device=DEVICE)
        
        # Annotate frame
        annotated_frame = results[0].plot()
        
        # Check for defects
        defect_found = any(int(box.cls.item()) in DEFECT_CLASSES 
                         for box in results[0].boxes)
        
        # Add status text
        status = "Defect Detected" if defect_found else "Quality OK"
        color = (0, 0, 255) if defect_found else (0, 255, 0)
        cv2.putText(annotated_frame, status, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Show confidence threshold
        cv2.putText(annotated_frame, f"Conf: {confidence:.2f}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Display output
        cv2.imshow('Manufacturing QC - Real Time', annotated_frame)

        # Keyboard controls
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('+'):
            confidence = min(1.0, confidence + 0.05)
        elif key == ord('-'):
            confidence = max(0.05, confidence - 0.05)

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# Run the system
# =====================================================
if __name__ == "__main__":
    print("Starting real-time quality inspection...")
    print("Press 'q' to quit, '+' to increase confidence, '-' to decrease")
    real_time_detection()