import os
import random
import cv2
import matplotlib.pyplot as plt

# Paths to your newly generated pseudo-labels
IMG_DIR = "dataset/pseudo_id/images"
LBL_DIR = "dataset/pseudo_id/labels"
NUM_SAMPLES = 5  # Number of random images to check

# Map YOLO class IDs back to readable names
CLASSES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}

def visualize_samples():
    # Get all generated images
    images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    if not images:
        print(f"No images found in {IMG_DIR}!")
        return

    # Randomly sample images
    samples = random.sample(images, min(NUM_SAMPLES, len(images)))

    # Set up a wide Matplotlib plot
    plt.figure(figsize=(20, 10))

    for i, img_name in enumerate(samples):
        img_path = os.path.join(IMG_DIR, img_name)
        lbl_path = os.path.join(LBL_DIR, img_name.replace('.jpg', '.txt'))

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Read YOLO labels if they exist
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cls_id, x_center, y_center, width, height = map(float, line.strip().split())

                    # Convert normalized YOLO format back to pixel coordinates
                    x_center, y_center = x_center * w, y_center * h
                    width, height = width * w, height * h

                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Draw bounding box (Green for hand, Blue for tool)
                    color = (0, 255, 0) if int(cls_id) == 0 else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                    # Draw label text
                    label = CLASSES.get(int(cls_id), str(int(cls_id)))
                    cv2.putText(img, label, (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Add to subplot
        plt.subplot(1, NUM_SAMPLES, i + 1)
        plt.imshow(img)
        plt.title(f"Sample {i + 1}")
        plt.axis('off')

    # Save to file instead of showing (safest for remote server execution)
    plt.tight_layout()
    output_path = 'sample_check.png'
    plt.savefig(output_path)
    print(f"Visualization saved to '{output_path}'. Open it in PyCharm to review your labels.")


if __name__ == '__main__':
    visualize_samples()