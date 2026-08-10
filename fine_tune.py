import os
from ultralytics import YOLO


def finetune_model():
    # 1. Load the BASE model you already trained
    # This applies the SSL step of refining the model with the new pseudo-labels
    model_path = 'weights/base_model.pt'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find {model_path}. Make sure you trained the base model first!")

    print(f"Loading baseline model from {model_path}...")
    model = YOLO(model_path)

    # 2. Retrain using the combined dataset (Original <100 images + Pseudo-labels)
    print("Starting fine-tuning process on combined dataset...")
    results = model.train(
        data='dataset_combined.yaml',
        epochs=60,
        imgsz=1280,
        rect=True,
        batch=4,
        patience=15, # Stop early if validation doesn't improve

        # --- Fine-tuning Hyperparameters ---
        lr0=0.001,  # Lower initial learning rate to preserve existing knowledge
        lrf=0.01,  # Final learning rate fraction

        # Keep augmentations active to prevent overfitting on the new pseudo-labels
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,

        project='runs/detect',
        name='finetuned_model',
        exist_ok=True
    )

    # 3. Save the newly fine-tuned model
    os.makedirs('weights', exist_ok=True)
    save_path = 'weights/finetuned_model.pt'
    model.save(save_path)
    print(f"\nFine-tuning complete! The stronger model is saved to '{save_path}'.")


if __name__ == '__main__':
    finetune_model()
