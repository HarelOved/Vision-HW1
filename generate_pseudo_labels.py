import os
import cv2
from ultralytics import YOLO

# Configuration
VIDEO_PATHS = [
    "/daatashare/HW1/id_video_data/4_2_24_B_2.mp4",
    "/datashare/HW1/id_video_data/20_2_24_1.mp4"
]

OUTPUT_IMG_DIR = "dataset/pseudo_id/images"
OUTPUT_LBL_DIR = "dataset/pseudo_id/labels"
MODEL_PATH = "weights/base_model.pt"

# Heuristic Selection Parameters
CONF_THRESHOLD = 0.65  # Keep predictions with high confidence score
FRAME_STRIDE = 10      # Process 1 frame every 10 frames to avoid duplicate samples

def generate_pseudo_labels():
    # Ensure output directories exist
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    # Load base model
    model = YOLO(MODEL_PATH)
    total_pseudo_frames = 0

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

            # Apply temporal sampling (frame stride)
            if frame_count % FRAME_STRIDE == 0:
                # Perform inference using high confidence filter
                results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
                boxes = results.boxes

                # Save frame only if at least one confident object was detected
                if len(boxes) > 0:
                    base_filename = f"vid{vid_idx}_frame_{frame_count}"
                    img_path = os.path.join(OUTPUT_IMG_DIR, f"{base_filename}.jpg")
                    txt_path = os.path.join(OUTPUT_LBL_DIR, f"{base_filename}.txt")

                    # Save extracted image frame
                    cv2.imwrite(img_path, frame)

                    # Write normalized YOLO annotation: (class x_center y_center width height)
                    with open(txt_path, "w") as f:
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            # xywhn returns normalized bounding box coordinates
                            xywhn = box.xywhn[0].tolist()
                            line = f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
                            f.write(line)

                    saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Saved {saved_count} pseudo-labeled frames from video {vid_idx + 1}.")
        total_pseudo_frames += saved_count

    print(f"\nCompleted! Generated {total_pseudo_frames} total pseudo-labeled training samples.")

if __name__ == '__main__':
    generate_pseudo_labels()