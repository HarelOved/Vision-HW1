import os
import shutil
import random


def create_hybrid_split():
    # --- 1. SET YOUR SOURCE DIRECTORIES HERE ---
    # Point these to where your 100 human-labeled images and labels currently live
    SOURCE_IMG_DIR = '/datashare/HW1/labeled_image_data/images/train'
    SOURCE_LBL_DIR = '/datashare/HW1/labeled_image_data/labels/train'

    # --- 2. SET THE DESTINATION DIRECTORY ---
    DEST_DIR = 'dataset_hybrid'

    # Create the new folder structure
    dirs_to_make = [
        f'{DEST_DIR}/combined_train/images',
        f'{DEST_DIR}/combined_train/labels',
        f'{DEST_DIR}/human_val/images',
        f'{DEST_DIR}/human_val/labels'
    ]
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

    # --- 3. GATHER AND SHUFFLE ---
    # Find all images (add other extensions if you use .jpeg)
    all_images = [f for f in os.listdir(SOURCE_IMG_DIR) if f.endswith(('.jpg', '.png'))]

    print(f"Found {len(all_images)} human-labeled images.")

    # Shuffle randomly (seed 42 ensures you get the same split if you run it twice)
    random.seed(42)
    random.shuffle(all_images)

    # Split exactly in half
    split_index = len(all_images) // 2
    train_anchors = all_images[:split_index]
    val_benchmarks = all_images[split_index:]

    # --- 4. COPY FILES HELPER FUNCTION ---
    def copy_split(image_list, split_name):
        for img_name in image_list:
            # Copy Image
            src_img = os.path.join(SOURCE_IMG_DIR, img_name)
            dst_img = os.path.join(DEST_DIR, split_name, 'images', img_name)
            shutil.copy(src_img, dst_img)

            # Find and Copy matching Label (.txt)
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            src_lbl = os.path.join(SOURCE_LBL_DIR, lbl_name)
            dst_lbl = os.path.join(DEST_DIR, split_name, 'labels', lbl_name)

            # Only copy if the label exists (handles background/empty images safely)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)

    # Execute the copying
    print("Copying 50 anchor images to combined_train...")
    copy_split(train_anchors, 'combined_train')

    print("Copying 50 benchmark images to human_val...")
    copy_split(val_benchmarks, 'human_val')

    # --- 5. GENERATE THE YAML FILE ---
    yaml_content = f"""# dataset_student_hybrid.yaml
# Train: Pseudo-labels (add them here!) + 50 human-labeled anchors
train: {DEST_DIR}/combined_train/images

# Val: 50 pristine human-labeled images ONLY
val: {DEST_DIR}/human_val/images

nc: 2
names: ['Needle_driver', 'Tweezers']
"""

    yaml_path = 'dataset_student_hybrid.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nSuccess! Generated folder structure in '{DEST_DIR}' and created '{yaml_path}'.")


if __name__ == '__main__':
    create_hybrid_split()