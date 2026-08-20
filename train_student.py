import os
from ultralytics import YOLO


def train_student_model():
    # 1. Initialize the Small "Student" model
    model = YOLO('yolo11l.pt')  # or 'yolo11s.pt' if you prefer

    # 2. Train on the COMBINED dataset
    results = model.train(
        data='dataset_student_hybrid.yaml',  # Must point to your Original + Pseudo-labels
        epochs=200,  # Far fewer epochs needed since the dataset is huge
        imgsz=1536,  # Keep the exact same resolution the Teacher used
        rect=True,

        # --- Memory & Compute ---
        # Because this is a Small model, your 24GB GPU can handle a much larger batch.
        # This drastically speeds up training and stabilizes gradient updates.
        batch=16,
        patience=0,

        # --- NO FREEZING ---
        # We want the Student to learn everything from scratch using the massive new dataset.
        # Notice we removed the freeze=10 argument completely.

        # --- Loss ---
        # cls=3.0,  # Keep the high penalty for class mix-ups

        # --- Augmentations ---
        mosaic=0.1,  # Keep disabled so thin tools aren't chopped in half
        mixup=0.0,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0.5,
        degrees=20.0,
        translate=0.2,
        scale=0.3,
        fliplr=0.5,
        shear=10,
        auto_augment='autoaugment',  # Use AutoAugment to generate more diverse samples

        project='runs/detect',
        name='student_model_1536_11s_hybrid_newest_2',
        exist_ok=True
    )

    print("\nStudent model training complete!")


if __name__ == '__main__':
    train_student_model()