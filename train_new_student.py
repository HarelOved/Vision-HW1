import os
from ultralytics import YOLO


def train_generation_3_model():
    # Initialize a fresh Medium model.
    # In Noisy Student architecture, it is often better to start with fresh COCO weights
    # rather than your previous Student's weights, so it doesn't get trapped in old biases.
    model = YOLO('yolo11s.pt')

    # Train with extreme augmentations for OOD generalization
    results = model.train(
        data='dataset_student_hybrid.yaml',  # Ensure this points to your new TTA-generated labels
        epochs=200,
        imgsz=1536,
        batch=16,
        patience=0,

        # --- CLASS IMBALANCE FIX ---
        cls=3.0,  # Crucial penalty to keep the model focused on minority classes

        # --- HEAVY AUGMENTATION (The "Noise") ---
        mosaic=1.0,  # 100% chance to combine 4 images into 1 (destroys background memorization)
        mixup=0.2,  # 20% chance to blend two images (forces robust feature extraction)
        degrees=45.0,  # Rotate tools up to 45 degrees (mimics weird wrist angles)
        translate=0.2,  # Shift the image by 20% (forces detection at the edges)
        scale=0.5,  # Scale images up and down by 50% (mimics camera zoom in OOD video)
        shear=5.0,  # Slightly stretch the image
        perspective=0.0005,  # 3D tilt effect (excellent for OOD camera angle changes)
        fliplr=0.5,  # 50% chance to flip horizontally
        flipud=0.2,  # 20% chance to flip vertically (great for overhead surgery views)

        # --- PHOTOMETRIC DISTORTION (Lighting/Color) ---
        hsv_h=0.015,  # Alter hue slightly
        hsv_s=0.7,  # Alter saturation heavily (mimics tissue/blood color changes)
        hsv_v=0.4,  # Alter brightness heavily (mimics harsh surgical lamp glare)

        # --- REGULARIZATION ---
        dropout=0.1,  # Randomly drop 10% of nodes to prevent overfitting to the ID video

        project='runs/detect',
        name='model_gen3_heavy_aug_11s',
        exist_ok=True
    )

    print("\nModel 3 (Heavy Augmentation) training complete!")


if __name__ == '__main__':
    train_generation_3_model()