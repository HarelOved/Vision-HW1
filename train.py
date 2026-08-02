import os
from ultralytics import YOLO


def train_base_model():
    # 1. Initialize with a pre-trained YOLO model (e.g., YOLOv8s)
    model = YOLO('yolov8s.pt')

    # 2. Train with boosted augmentations for heavy regularization
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,

        # --- Augmentation Hyperparameters ---
        hsv_h=0.015,  # Hue jitter
        hsv_s=0.7,  # Saturation jitter
        hsv_v=0.4,  # Brightness jitter
        degrees=15.0,  # Image rotation (+/- deg)
        translate=0.1,  # Translation (+/- fraction)
        scale=0.5,  # Scaling (+/- gain)
        shear=2.0,  # Shear (+/- deg)
        fliplr=0.5,  # Horizontal flip (natural for surgical hands/tools)
        mosaic=1.0,  # Mosaic augmentation (combines 4 images)
        mixup=0.1,  # Mixup augmentation

        project='runs/detect',
        name='base_model',
        exist_ok=True
    )
    
    metrics = model.val()

    # 3. Save final base model weights to a clean path
    os.makedirs('weights', exist_ok=True)
    model.save('weights/base_model.pt')
    print("\nBase model training complete! Model saved to 'weights/base_model.pt'.")


if __name__ == '__main__':
    train_base_model()