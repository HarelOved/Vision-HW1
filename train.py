from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8 model
# 'yolov8n.pt' is the Nano version (lightest and fastest).
# You can also use 'yolov8s.pt' (Small) or 'yolov8m.pt' (Medium).
model = YOLO('yolov8n.pt')

# 2. Train the model
# The train method uses the dataset configuration to start the training process.
results = model.train(
    data='dataset.yaml',  # Path to the configuration file
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size (640 is standard for YOLO)
    batch=16,  # Batch size (decrease if running out of GPU memory)
    name='surgical_tools',  # Name of the folder where results will be saved
    device='0',  # Use GPU ('0'). Change to 'cpu' if no GPU is available
    patience=10,  # Early stopping if validation metrics do not improve for 10 epochs

    # Data augmentation settings to overcome the limited dataset size:
    fliplr=0.5,  # Probability of horizontal flip
    mosaic=1.0  # Mosaic augmentation (combines 4 images into 1 to improve generalization)
)

# 3. Evaluate the model
# After training, evaluate the model's performance (mAP) on the validation set.
metrics = model.val()

print("Training completed successfully!")

model.save('surgical_tools_final.pt')  # Save the final trained model