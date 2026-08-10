import os
from ultralytics import YOLO


def train_student_model():
    # 1. Initialize the Small "Student" model
    model = YOLO('yolo26m.pt')  # or 'yolo11s.pt' if you prefer

    # 2. Train on the COMBINED dataset
    results = model.train(
        data='dataset_combined.yaml',  # Must point to your Original + Pseudo-labels
        epochs=120,  # Far fewer epochs needed since the dataset is huge
        imgsz=1536,  # Keep the exact same resolution the Teacher used
        rect=True,

        # --- Memory & Compute ---
        # Because this is a Small model, your 24GB GPU can handle a much larger batch.
        # This drastically speeds up training and stabilizes gradient updates.
        batch=16,
        patience=35,

        # --- NO FREEZING ---
        # We want the Student to learn everything from scratch using the massive new dataset.
        # Notice we removed the freeze=10 argument completely.

        # --- Loss ---
        cls=3.0,  # Keep the high penalty for class mix-ups

        # --- Augmentations ---
        mosaic=0.0,  # Keep disabled so thin tools aren't chopped in half
        mixup=0.0,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,

        project='runs/detect',
        name='student_model_1536_26',
        exist_ok=True
    )

    print("\nStudent model training complete!")


if __name__ == '__main__':
    train_student_model()