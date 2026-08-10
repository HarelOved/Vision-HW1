from ultralytics import YOLO


def process_ood_video():
    # Load your trained Student model
    model = YOLO('/home/student/Harel_HW1/runs/detect/runs/detect/base_model_1536_yolo26x_new/weights/best.pt')

    # Run tracking prediction on the video
    results = model.track(
        source='path/to/your/ood_video.mp4',  # Update this path!

        # --- FIX 1: Double Boxes ---
        iou=0.3,  # Strict NMS: If two boxes overlap by 30%, delete the weaker one

        # --- FIX 2: Misclassifications ---
        conf=0.45,  # Ignore any weak, random guesses under 45% confidence
        tracker="botsort.yaml",  # Forces temporal consistency across frames

        # --- Visualization ---
        show=True,  # Watch the output live
        save=True,  # Save the final fixed video
        project='runs/detect',
        name='ood_video_fixed'
    )

    print("\nVideo processing complete! Saved to runs/detect/ood_video_fixed")


if __name__ == '__main__':
    process_ood_video()