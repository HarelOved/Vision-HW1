import os
from ultralytics import YOLO


def train_cooldown_box_refinement():
    # 1. Start with your heavily augmented model that has great classification
    model = YOLO('/home/student/Harel_HW1/runs/detect/runs/detect/model_gen3_heavy_aug_11s/weights/best.pt')

    # 2. Train on the EXACT same ID dataset you just used
    results = model.train(
        data='dataset.yaml',  # Your ID pseudo-labels
        epochs=30,  # Very short run! We just want to tighten boxes.
        imgsz=1536,
        batch=8,

        # --- THE FIX: DISABLE GEOMETRIC DISTORTIONS ---
        mosaic=0.0,  # Turn off mosaic so tools are whole
        mixup=0.0,
        perspective=0.0,
        shear=0.0,
        degrees=0.0,  # No rotation, let it see the true edges
        scale=0.0,

        # --- KEEP LIGHTING AUGMENTATIONS ---
        hsv_h=0.015,
        hsv_s=0.7,  # Keep saturation shifts
        hsv_v=0.6,  # KEEP BRIGHTNESS SHIFTS so it survives the OOD video!

        # --- SHIFT LOSS FOCUS TO BOUNDING BOXES ---
        box=7.5,  # Massive penalty for loose boxes (default is 7.5)
        dfl=2.5,  # Forces pixel-perfect edge detection (default is 1.5)
        cls=1.0,  # Lower this back down; it already knows the classes

        # --- LOWER LEARNING RATE ---
        lr0=0.0001,  # Tiny learning rate so we don't ruin the classification weights
        lrf=0.1,
        project='runs/detect',
        name='model_final_final_box_cooldown',
        exist_ok=True
    )

    print("\nBox Cooldown training complete!")


if __name__ == '__main__':
    train_cooldown_box_refinement()