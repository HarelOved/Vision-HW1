import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def process_video_with_tracking(input_video_path, model_path, output_video_path, output_txt_path, conf_thresh=0.5):
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Video not found: {input_video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    print(f"Loading fine-tuned model from: {model_path}")
    model = YOLO(model_path)
    class_names = model.names

    np.random.seed(42)
    class_colors = {
        cls_id: tuple(np.random.randint(0, 255, 3).tolist())
        for cls_id in class_names.keys()
    }

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    print("\nProcessing video with temporal tracking...")

    with open(output_txt_path, "w") as f_out:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Using model.track with persist=True maintains object IDs across frames (prevents flickering)
            results = model.track(frame, imgsz=1536, conf=conf_thresh, persist=True, verbose=False, augment=True, iou=0.25)[0]
            boxes = results.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x_c, y_c, w_norm, h_norm = box.xywhn[0].tolist()
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())

                    f_out.write(
                        f"Frame {frame_idx}: {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f} {conf:.4f} {cls_id}\n")

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_name = class_names.get(cls_id, f"Class {cls_id}")
                    color = class_colors.get(cls_id, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
                    label_text = f"{class_name} {conf:.2f}"

                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    bg_y1 = max(0, y1 - text_h - 10)
                    bg_y2 = max(text_h + 10, y1)

                    cv2.rectangle(frame, (x1, bg_y1), (x1 + text_w + 10, bg_y2), color, cv2.FILLED)
                    cv2.putText(frame, label_text, (x1 + 5, bg_y2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)

            if frame_idx % 100 == 0 or frame_idx == total_frames:
                print(f"Processed frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    print(f"\nFinished processing!")
    print(f"Annotated video saved to: {output_video_path}")
    print(f"YOLO labels saved to: {output_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/datashare/HW1/ood_video_data/4_2_24_A_1.mp4")
    #parser.add_argument("--input", type=str, default="/datashare/HW1/id_video_data/4_2_24_B_2.mp4")
    parser.add_argument("--weights", type=str, default="/home/student/Harel_HW1/runs/detect/runs/detect/model_final_final_box_cooldown/weights/best.pt")
    parser.add_argument("--output_vid", type=str, default="final_aug_ood_prediction_output.mp4")
    parser.add_argument("--output_txt", type=str, default="ood_yolo_labels.txt")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    process_video_with_tracking(args.input, args.weights, args.output_vid, args.output_txt, args.conf)