import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import random

# Base directories
# IMAGES_DIR = "/datashare/HW1/labeled_image_data/images/train"
# LABELS_DIR = "/datashare/HW1/labeled_image_data/labels/train"

IMAGES_DIR = "dataset_v5/pseudo_id/images"
LABELS_DIR = "dataset_v5/pseudo_id/labels"


# Map numeric classes to names.
CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_Driver"}
COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}  # Red, Green, Blue in RGB


def load_yolo_labels(labels_dir, images_dir):
    """Automatically parses all YOLO .txt label files in the directory."""
    data_dict = {}
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    for label_path in label_files:
        base_name = os.path.splitext(os.path.basename(label_path))[0]

        # Match the label file to its corresponding image (.jpg or .png)
        img_filename = f"{base_name}.jpg"
        if not os.path.exists(os.path.join(images_dir, img_filename)):
            if os.path.exists(os.path.join(images_dir, f"{base_name}.png")):
                img_filename = f"{base_name}.png"

        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    labels.append([class_id, x_c, y_c, w, h])

        # Only add to dictionary if the file wasn't empty
        if labels:
            data_dict[img_filename] = labels

    return data_dict


def yolo_to_bbox(x_center, y_center, w, h, img_w, img_h):
    """Convert YOLO normalized coordinates to standard pixel bounding box coordinates."""
    x_min = int((x_center - w / 2) * img_w)
    y_min = int((y_center - h / 2) * img_h)
    x_max = int((x_center + w / 2) * img_w)
    y_max = int((y_center + h / 2) * img_h)
    return x_min, y_min, x_max, y_max


def visualize_images(data_dict, image_dir, num_samples=3):
    """Draw bounding boxes on a random sample of images."""
    # Randomly sample images so Matplotlib doesn't crash trying to render 100 images
    sample_keys = random.sample(list(data_dict.keys()), min(num_samples, len(data_dict)))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    if len(sample_keys) == 1:
        axes = [axes]  # Ensure iterable if only 1 image

    idx = 0
    for ax in axes:
        filename = sample_keys[idx]
        labels = data_dict[filename]
        img_path = os.path.join(image_dir, filename)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print(f"Warning: {filename} not found locally. Using blank canvas.")
            img = np.ones((720, 1280, 3), dtype=np.uint8) * 200

        img_h, img_w, _ = img.shape

        for label in labels:
            cls_id = int(label[0])
            x_c, y_c, w, h = label[1:]

            x_min, y_min, x_max, y_max = yolo_to_bbox(x_c, y_c, w, h, img_w, img_h)
            color = COLORS.get(cls_id, (255, 255, 255))

            # Draw Rectangle and Label
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
            label_text = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            cv2.putText(img, label_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(filename, fontsize=8)
        idx += 1

    plt.tight_layout()
    plt.show()


def analyze_distributions(data_dict):
    """Analyze and plot class distributions and bounding box sizes."""
    all_labels = []
    for filename, labels in data_dict.items():
        for label in labels:
            all_labels.append({
                "filename": filename,
                "class_id": int(label[0]),
                "x_center": label[1],
                "y_center": label[2],
                "width": label[3],
                "height": label[4],
                "area": label[3] * label[4]  # Normalized area
            })

    if not all_labels:
        print("No labels found to analyze.")
        return

    df = pd.DataFrame(all_labels)
    df['class_name'] = df['class_id'].map(CLASS_NAMES)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Class Distribution
    class_counts = df['class_name'].value_counts()
    axes[0].bar(class_counts.index, class_counts.values, color=['blue', 'orange', 'green'])
    axes[0].set_title("Class Frequency Distribution")
    axes[0].set_ylabel("Number of Bounding Boxes")

    # 2. Bounding Box Area Distribution (Size)
    for cls in df['class_id'].unique():
        subset = df[df['class_id'] == cls]
        axes[1].hist(subset['area'], alpha=0.5, bins=10, label=CLASS_NAMES.get(cls))
    axes[1].set_title("Bounding Box Area")
    axes[1].set_xlabel("Area")
    axes[1].legend()

    # 3. Spatial Distribution
    for cls in df['class_id'].unique():
        subset = df[df['class_id'] == cls]
        axes[2].scatter(subset['x_center'], subset['y_center'], alpha=0.7, label=CLASS_NAMES.get(cls))

    axes[2].set_title("Spatial Distribution of Centers")
    axes[2].set_xlabel("X Center")
    axes[2].set_ylabel("Y Center")
    axes[2].invert_yaxis()
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# --- Execution Flow ---
print("Loading labels from server...")
full_dataset = load_yolo_labels(LABELS_DIR, IMAGES_DIR)
print(f"Successfully loaded labels for {len(full_dataset)} images.")

print("Visualizing a random sample of images")
visualize_images(full_dataset, IMAGES_DIR)

print("Analyzing data distribution across the full dataset...")
analyze_distributions(full_dataset)