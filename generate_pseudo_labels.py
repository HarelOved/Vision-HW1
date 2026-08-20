import os
import cv2
import collections
from ultralytics import YOLO

# Configuration
VIDEO_PATHS = [
    "/datashare/HW1/id_video_data/4_2_24_B_2.mp4",
    "/datashare/HW1/id_video_data/20_2_24_1.mp4"
]

OUTPUT_IMG_DIR = "dataset_v5/pseudo_id/images"
OUTPUT_LBL_DIR = "dataset_v5/pseudo_id/labels"
OUTPUT_SAMPLE_DIR = "dataset_v5/pseudo_id/samples"
#MODEL_PATH = "/home/student/Harel_HW1/runs/detect/runs/detect/base_model_1536_yolo26x_new/weights/best.pt"
MODEL_PATH = "/home/student/Harel_HW1/runs/detect/runs/detect/student_model_1536_11s_hybrid_newest_2/weights/best.pt"


# --- THE FIX: CLASS-SPECIFIC THRESHOLDS ---
# 0: Empty (Low threshold because they are hard to distinguish from tweezers)
# 1: Tweezers (High threshold because the model is already over-predicting them)
# 2: Needle_driver (Medium threshold to boost their numbers)
CONF_THRESHOLDS = {
    0: 0.6,
    1: 0.35,
    2: 0.35
}

FRAME_STRIDE = 10
IMGSZ = 1536
SAMPLES_PER_CLASS = 10


def generate_pseudo_labels():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SAMPLE_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    class_names = model.names

    total_pseudo_frames = 0
    class_counts = collections.Counter()
    samples_saved = collections.defaultdict(int)

    for vid_idx, video_path in enumerate(VIDEO_PATHS):
        if not os.path.exists(video_path):
            print(f"Warning: Video path not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0

        print(f"Processing Video ({vid_idx + 1}/{len(VIDEO_PATHS)}): {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_STRIDE == 0:
                # 1. Base Inference: Catch everything above 0.20
                results = model.predict(frame, imgsz=IMGSZ, conf=0.20, verbose=False)[0]

                # 2. Custom Filtering: Apply our dictionary thresholds
                valid_boxes = []
                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    conf_score = float(box.conf[0].item())

                    # Check if the box confidence beats the specific threshold for its class
                    target_threshold = CONF_THRESHOLDS.get(cls_id, 0.50)
                    if conf_score >= target_threshold:
                        valid_boxes.append(box)

                # 3. Save only if valid boxes survived the filter
                if len(valid_boxes) > 0:
                    base_filename = f"vid{vid_idx}_frame_{frame_count}"
                    img_path = os.path.join(OUTPUT_IMG_DIR, f"{base_filename}.jpg")
                    txt_path = os.path.join(OUTPUT_LBL_DIR, f"{base_filename}.txt")

                    cv2.imwrite(img_path, frame)

                    frame_classes = set()

                    # Write YOLO annotations for the VALID boxes only
                    with open(txt_path, "w") as f:
                        for box in valid_boxes:
                            cls_id = int(box.cls[0].item())
                            class_counts[cls_id] += 1
                            frame_classes.add(cls_id)

                            xywhn = box.xywhn[0].tolist()
                            line = f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
                            f.write(line)

                    # 4. Custom Drawing for Samples (so you don't see the deleted boxes)
                    for cls_id in frame_classes:
                        if samples_saved[cls_id] < SAMPLES_PER_CLASS:
                            samples_saved[cls_id] += 1
                            cls_name = class_names.get(cls_id, f"class_{cls_id}")
                            sample_path = os.path.join(
                                OUTPUT_SAMPLE_DIR,
                                f"sample_{cls_name}_{samples_saved[cls_id]}_{base_filename}.jpg"
                            )

                            # Draw boxes manually onto the frame copy
                            annotated_frame = frame.copy()
                            for v_box in valid_boxes:
                                x1, y1, x2, y2 = map(int, v_box.xyxy[0].tolist())
                                c_id = int(v_box.cls[0].item())
                                c_conf = float(v_box.conf[0].item())
                                label = f"{class_names.get(c_id, 'Unknown')} {c_conf:.2f}"

                                # Draw Rectangle and Label
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2)

                            cv2.imwrite(sample_path, annotated_frame)

                    saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Saved {saved_count} pseudo-labeled frames from video {vid_idx + 1}.")
        total_pseudo_frames += saved_count

    print("\n" + "=" * 50)
    print("          PSEUDO-LABELING SUMMARY          ")
    print("=" * 50)
    print(f"Total Pseudo-Labeled Frames Generated: {total_pseudo_frames}")
    print("\nClass Instance Breakdown:")
    for cls_id, name in class_names.items():
        count = class_counts[cls_id]
        print(f"  • Class {cls_id} ({name}): {count} total instances")
    print(f"\nAnnotated inspection samples saved to: '{OUTPUT_SAMPLE_DIR}'")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    generate_pseudo_labels()