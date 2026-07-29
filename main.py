import cv2
import matplotlib.pyplot as plt

# Define the exact path to your image
img_path = "/datashare/HW1/labeled_image_data/images/train/245d16f4-frame_2683.jpg"

# Read the image using OpenCV
img = cv2.imread(img_path)

# Check if the image was successfully loaded
if img is not None:
    # OpenCV loads images in BGR format, but Matplotlib expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a plot and display the image
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide the axes for a cleaner look
    plt.title("245d16f4-frame_2683.jpg")
    plt.show()
else:
    print(f"Error: Could not find or open the image at {img_path}. Double-check the file path.")