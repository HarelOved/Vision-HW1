import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 1. Define the sample data provided in the prompt
# Dictionary mapping filenames to their YOLO labels (class, x_center, y_center, width, height)
sample_data = {
    "d461d90b-frame_0299.jpg": [
        [2, 0.4141009852216749, 0.5238095705951785, 0.2703201970443349, 0.18938140615501772],
        [1, 0.5637315270935959, 0.41707722248351997, 0.10775862068965517, 0.32621774988791313]
    ],
    "d3410f1b-output_0186.jpg": [
        [2, 0.29710591133004927, 0.4121510673234811, 0.28386699507389157, 0.528735632183908],
        [1, 0.5458743842364532, 0.4422550629447181, 0.20012315270935957, 0.45539135194307606]
    ],
    "db41f653-output_0103.jpg": [
        [0, 0.19396551724137928, 0.09578544061302682, 0.13669950738916253, 0.19157088122605365],
        [1, 0.44550469505263435, 0.3103446872292423, 0.1176112995499032, 0.3689110634178144],
        [0, 0.911022120091635, 0.36070066757982894, 0.10652718838815663, 0.18500286772670782]
    ],
    "dc469e1a-frame_0895.jpg": [
        [2, 0.40578817733990147, 0.6392993979200875, 0.2955665024630542, 0.1663929939792009],
        [1, 0.6009852779576851, 0.48056918304714263, 0.10591121748364413, 0.2933772332488989]
    ],
    "cfcbe252-output_0178.jpg": [
        [2, 0.47444581280788173, 0.7493158182813354, 0.2789408866995074, 0.18938149972632728],
        [0, 0.5132389162561576, 0.6081007115489874, 0.166871921182266, 0.3295019157088122]
    ]
}

# Optional: Map numeric classes to names if you know them.
# Based on common surgical datasets, this might be:
CLASS_NAMES = {0: "0", 1: "1", 2: "2"}
COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}  # Red, Green, Blue in RGB


def yolo_to_bbox(x_center, y_center, w, h, img_w, img_h):
    """Convert YOLO normalized coordinates to standard pixel bounding box coordinates."""
    x_min = int((x_center - w / 2) * img_w)
    y_min = int((y_center - h / 2) * img_h)
    x_max = int((x_center + w / 2) * img_w)
    y_max = int((y_center + h / 2) * img_h)
    return x_min, y_min, x_max, y_max


def visualize_images(data_dict, image_dir="//datashare//HW1//labeled_image_data//images//train"):
    """Draw bounding boxes on images and plot them."""
    fig, axes = plt.subplots(1, len(data_dict), figsize=(20, 5))

    for ax, (filename, labels) in zip(axes, data_dict.items()):
        img_path = os.path.join(image_dir, filename)

        # In a real environment, read the actual image.
        # Here we create a dummy image if the file isn't found locally.
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

    plt.tight_layout()
    plt.show()


def analyze_distributions(data_dict):
    """Analyze and plot class distributions and bounding box sizes."""
    # Flatten the data into a Pandas DataFrame
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

    df = pd.DataFrame(all_labels)
    df['class_name'] = df['class_id'].map(CLASS_NAMES)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Class Distribution
    class_counts = df['class_name'].value_counts()
    axes[0].bar(class_counts.index, class_counts.values, color=['blue', 'green', 'red'])
    axes[0].set_title("Class Frequency Distribution")
    axes[0].set_ylabel("Number of Bounding Boxes")

    # 2. Bounding Box Area Distribution (Size)
    for cls in df['class_id'].unique():
        subset = df[df['class_id'] == cls]
        axes[1].hist(subset['area'], alpha=0.5, bins=10, label=CLASS_NAMES.get(cls))
    axes[1].set_title("Bounding Box Area (Normalized Size)")
    axes[1].set_xlabel("Area (Width * Height)")
    axes[1].legend()

    # 3. Spatial Distribution (Where do objects appear on screen?)
    for cls in df['class_id'].unique():
        subset = df[df['class_id'] == cls]
        axes[2].scatter(subset['x_center'], subset['y_center'], alpha=0.7, label=CLASS_NAMES.get(cls))

    axes[2].set_title("Spatial Distribution of Centers")
    axes[2].set_xlabel("X Center")
    axes[2].set_ylabel("Y Center")
    axes[2].invert_yaxis()  # Image coordinates start at top-left
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# Run the EDA functions
print("Visualizing Bounding Boxes...")
visualize_images(sample_data)

print("Analyzing Data Distribution...")
analyze_distributions(sample_data)