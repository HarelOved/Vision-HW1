import os
from ultralytics import YOLO


def train_base_model():
    # Load the Extra Large YOLO26 model
    model = YOLO('yolo11s.pt')

    results = model.train(
        data='dataset.yaml',
        epochs=400,
        imgsz=1536,
        rect=True,

        # --- Memory & Compute Settings ---
        # 24GB VRAM can typically handle batch=4 or batch=8 at 1280px for the 'x' model.
        # If you get a CUDA Out of Memory (OOM) error, drop this to 2 or 4.
        batch=16,

        patience=0,
        # freeze=10,

        # --- Fix for Needle Drivers (Class Imbalance/Difficulty) ---
        cls=3.0,
        # fl_gamma=1.5,

        # --- Augmentations ---
        lr0=0.001,
        lrf=0.01,
        # --- FIX 4: Safe Augmentations for Thin Tools ---
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        degrees=20.0,
        translate=0.3,
        scale=0.5,
        shear=10,
        fliplr=0.5,
        mosaic=0.0,
        erasing=0.0,
        auto_augment=None,

        project='runs/detect',
        name='base_model_1536_yolo11s_newest',
        exist_ok=True
    )

    os.makedirs('weights', exist_ok=True)
    model.save('weights/base_model.pt')
    print("\nBase model training complete! Saved to 'weights/base_model.pt'.")


def evaluate_both_splits():
    # 1. Load your newly trained model
    model = YOLO('weights/teacher_model.pt')

    print("=== Evaluating on the TRAINING set ===")
    # Setting split='train' forces the evaluator to look at your train images
    train_metrics = model.val(data='dataset.yaml', split='train')

    print("\n=== Evaluating on the VALIDATION set ===")
    # Default behavior (looks at val images)
    val_metrics = model.val(data='dataset.yaml', split='val')

    # You can print specific metrics if you want to log them for your report
    print(f"\nFinal Train mAP50: {train_metrics.box.map50:.3f}")
    print(f"Final Val mAP50:   {val_metrics.box.map50:.3f}")


if __name__ == '__main__':
    train_base_model()
    evaluate_both_splits()

