import cv2
import numpy as np
import os

def resize_and_center_on_canvas(image_path, output_path, canvas_size=28, margin=4):
    # Read the cropped image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was successfully loaded
    if img is None:
        print(f"Error: Unable to load the image from '{image_path}'. Skipping...")
        return

    # Get dimensions of the image
    h, w = img.shape

    # Calculate the available space for the letter (after accounting for margin)
    available_size = canvas_size - margin * 2

    # Calculate the scaling factor to fit the image within the available space
    scale_factor = min(available_size / w, available_size / h)

    # Resize the image while keeping the aspect ratio
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank white canvas (255 for white pixels)
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

    # Calculate offsets to center the resized image on the canvas, including margin
    x_offset = (canvas_size - new_w) // 2
    y_offset = (canvas_size - new_h) // 2

    # Place the resized image onto the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    # Save the result
    cv2.imwrite(output_path, canvas)
    print(f"Resized image saved as {output_path}")

def process_images_in_directory(input_dir, output_dir, canvas_size=28, margin=4):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)

            # Resize and center the image
            resize_and_center_on_canvas(input_image_path, output_image_path, canvas_size, margin)

# Specify the input directory and the output directory
input_directory = "/Users/cnipper/vsworkplace/.venv/handwritten/!words/letters"  # Replace with the actual directory containing your cropped images
output_directory = "/Users/cnipper/vsworkplace/.venv/handwritten/!words/letters"  # Directory to save resized images

# Process all images in the directory with added whitespace
process_images_in_directory(input_directory, output_directory, canvas_size=28, margin=4)
