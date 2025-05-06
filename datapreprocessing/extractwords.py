import cv2
import numpy as np
import os
import glob

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(img, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 
                                   11, 2)
    
    # Dilation to connect letters
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(thresh, kernel, iterations=1)

    return img, dilated_img

def merge_contours(bounding_boxes, space_threshold):
    merged_boxes = []
    for box in bounding_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            x, y, w, h = box
            prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
            # Check horizontal distance between bounding boxes
            if x - (prev_x + prev_w) < space_threshold:
                # If the boxes are close horizontally, merge them
                new_x = min(prev_x, x)
                new_y = min(prev_y, y)
                new_w = max(prev_x + prev_w, x + w) - new_x
                new_h = max(prev_y + prev_h, y + h) - new_y
                merged_boxes[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged_boxes.append(box)
    return merged_boxes

def extract_words(image_path, output_directory, space_threshold):
    # Preprocess the image
    original_img, dilated_img = preprocess_image(image_path)

    # Find contours in the image
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    delete_all_files(output_directory)
    word_number = 0

    # Sort contours from left to right based on x-coordinate
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    # Merge contours based on horizontal spacing
    merged_boxes = merge_contours(bounding_boxes, space_threshold)

    # Loop over merged bounding boxes and save each word
    for (x, y, w, h) in merged_boxes:
        word_number += 1
        word_img = original_img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_directory, f'word{word_number:02d}.png'), word_img)

    print(f"Extracted {word_number} words and saved them in {output_directory}")

def extract_letters(word_image_path, output_directory):
    # Preprocess the word image
    original_img, dilated_img = preprocess_image(word_image_path)

    # Find contours in the word image
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    delete_all_files(output_directory)
    letter_number = 0  # Start from 0

    # Sort contours from left to right based on the x-coordinate of their bounding boxes
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Filter out small contours
    bounding_boxes = [b for b in bounding_boxes if b[2] > 3 and b[3] > 3]  # Minimum width and height

    # Sort bounding boxes by the x-coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    last_x_end = None  # To track the end of the last bounding box

    # Loop over each bounding box and save each letter as an image
    for (x, y, w, h) in bounding_boxes:
        # Calculate padding
        padding_x = 2  # Padding on the sides (horizontal)
        padding_y = 2  # Padding on the top and bottom (vertical)

        # Crop the letter from the original image with padding
        letter_img = original_img[max(0, y-padding_y):y+h+padding_y, max(0, x-padding_x):x+w+padding_x]

        # Check if this is a new letter based on horizontal space
        if last_x_end is None or (x - last_x_end > padding_x):  # Using padding_x as a dynamic threshold
            # Increment the letter number
            letter_number += 1

            # Ensure we only save non-empty letter images
            if letter_img.size > 0:
                # Save the letter image to the output directory
                cv2.imwrite(os.path.join(output_directory, f'letter{letter_number:02d}.png'), letter_img)

        # Update the end position
        last_x_end = x + w

    print(f"Extracted {letter_number} letters and saved them in {output_directory}")


def scaleLetterImages(imagePath, factor):
    # Load the image
    image = cv2.imread(imagePath)

    # Scale up the image by a factor (e.g., 2.0)
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)

    # Resize the image
    scaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Save the scaled image
    cv2.imwrite(imagePath, scaled_image)

def delete_all_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    # Define the path to the handwritten sentences image and output directory
    examples_directory = "/Users/cnipper/vsworkplace/.venv/handwritten/!examples"  # Update with your image path
    words_directory = "/Users/cnipper/vsworkplace/.venv/handwritten/!examples/words"  # The directory to save the individual word images
    letters_directory = "/Users/cnipper/vsworkplace/.venv/handwritten/!examples/words/letters"
    outputstring = ''

    # Extract words from the image and save them
    space_threshold = 10 # Adjust this value to change the required horizontal space between words
    files = glob.glob(os.path.join(examples_directory, '*.png'))
    files.sort()
    for image in files:
        extract_words(image, words_directory, space_threshold)
        words = glob.glob(os.path.join(words_directory, '*.png'))
        words.sort()
        for word in words:
            scaleLetterImages(word, 4)
            extract_letters(word, letters_directory)
            i = input()
    
    delete_all_files(words_directory)
    delete_all_files(letters_directory)
    os.system('clear')