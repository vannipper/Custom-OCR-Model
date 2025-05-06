import cv2
import os

# Function to crop images around a character based on contours
def crop_character_image(image_path, output_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding to distinguish the character
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours around the character
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return (likely an empty image)
    if len(contours) == 0:
        print(f"No character found in {image_path}")
        return

    # Find the bounding rectangle for the largest contour (character)
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Crop the image to the bounding box
    cropped_image = img[y:y+h, x:x+w]

    # Save the cropped image to the specified output path
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved as {output_path}")

# Main function to crop all images in a directory
def process_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through all image files in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f"{filename}")
            
            # Crop the image and save the result
            crop_character_image(image_path, output_path)

# Specify input and output directories
input_directory = "/Users/cnipper/vsworkplace/.venv/handwritten/!words/letters"  # Replace with your input directory path
output_directory = "/Users/cnipper/vsworkplace/.venv/handwritten/!words/letters"  # Replace with your desired output directory path

# Run the cropping function on all images in the directory
process_directory(input_directory, output_directory)
