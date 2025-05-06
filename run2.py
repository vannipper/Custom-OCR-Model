import cv2
import numpy as np
import os
import tensorflow as tf

def extract_words(image_path, output_directory, space_threshold): # extracts words from image of sentence
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Preprocess the image
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the image

    if not os.path.exists(output_directory): # Ensure the output directory exists
        os.makedirs(output_directory)

    delete_all_files(output_directory)
    word_number = 0

    bounding_boxes = [cv2.boundingRect(c) for c in contours] # Sort contours from left to right based on x-coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    
    merged_boxes = [] # Merge contours based on horizontal spacing
    for box in bounding_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            x, y, w, h = box
            prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
            if x - (prev_x + prev_w) < space_threshold: # Check horizontal distance between bounding boxes
                new_x = min(prev_x, x) # If the boxes are close horizontally, merge them
                new_y = min(prev_y, y)
                new_w = max(prev_x + prev_w, x + w) - new_x
                new_h = max(prev_y + prev_h, y + h) - new_y
                merged_boxes[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged_boxes.append(box)

    for (x, y, w, h) in merged_boxes: # Loop over merged bounding boxes and save each word
        word_number += 1
        word_img = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_directory, f'word{word_number:02d}.png'), word_img)

    # print(f"Extracted {word_number} words and saved them in {output_directory}")

def extract_letters(image_path, output_directory, space_threshold):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
    #img = scaleImage(img, 4)  # Scale image for better processing
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(thresh, kernel, iterations=1)
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not os.path.exists(output_directory):  # Ensure the output directory exists
        os.makedirs(output_directory)

    delete_all_files(output_directory)
    letter_number = 0

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])  # Sort by x-coordinate

    merged_boxes = []  # Merge contours based on horizontal spacing
    for box in bounding_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            x, y, w, h = box
            prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
            # Check if boxes are too close together
            if x - (prev_x + prev_w) < space_threshold:
                new_x = min(prev_x, x)
                new_y = min(prev_y, y)
                new_w = max(prev_x + prev_w, x + w) - new_x
                new_h = max(prev_y + prev_h, y + h) - new_y
                merged_boxes[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged_boxes.append(box)

    # Filter and save each letter
    for (x, y, w, h) in merged_boxes:
        # Filter out small boxes based on height and width
        if w > 5 and h > 10:  # Adjust these thresholds as needed
            letter_number += 1
            letter_img = img[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_directory, f'letter{letter_number:02d}.png'), letter_img)

    # print(f"Extracted {letter_number} letters and saved them in {output_directory}")

def resize(input_dir, output_dir, canvas_size, margin): # formats extracted letter images for insertion to NN
    if not os.path.exists(output_dir): # Ensure the output directory exists
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir): # Loop over all images in the input directory
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None: # Check if the image was successfully loaded
                print(f"Error: Unable to load the image from '{image_path}'. Skipping...")
                return

            h, w = img.shape # Get dimensions of the image

            available_size = canvas_size - margin * 2
            scale_factor = min(available_size / w, available_size / h)

            new_w = int(w * scale_factor) # Resize the image while keeping the aspect ratio
            new_h = int(h * scale_factor)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
            x_offset = (canvas_size - new_w) // 2
            y_offset = (canvas_size - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img # Place the resized image onto the canvas

            cv2.imwrite(output_path, canvas)

def delete_all_files(directory): # deletes all files in a directory which is given as a parameter

    if not os.path.isdir(directory):
        raise ValueError(f"The specified path '{directory}' is not a valid directory.")
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def scaleImage(image, factor): # takes an image object, returns it scaled

    # Scale up the image by a factor (e.g., 2.0)
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)

    # Resize the image
    scaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    return(scaled_image)

def viewImage(image_path, zoom_factor, window_name='letterImage', delay=0): # views a scaled image given its image path and a scale factor
    image = cv2.imread(image_path) # Load the image from the provided file path
    
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return
    
    image = scaleImage(image, zoom_factor)

    cv2.imshow(window_name, image) # Display the zoomed-in image
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def calculateAccuracy(outputstr):
    correctstr, correct, total = input('Please input the expected string: '), 0, 0
    for i, char in enumerate(correctstr.lower()):
        try:
            if char == outputstr[i]:
                correct += 1
        except (IndexError):
            pass
        total += 1
    return correct/total

if __name__ == "__main__":

    examples_directory = "/Users/cnipper/vsworkplace/.venv/handwritten/!examples" # Directories
    words_directory = os.path.join(examples_directory, 'words')
    letters_directory = os.path.join(words_directory, 'letters')

    modelname = input('Enter model name: ') # Load CNN
    model = tf.keras.models.load_model(f'models/{modelname}.keras')

    for sentence in sorted([f for f in os.listdir(examples_directory) if f.endswith('.png')]): # Parse through example data by sentence, then word, then letter
        outputstring = ''
        extract_words(os.path.join(examples_directory, sentence), words_directory, 10)
        for word in sorted([f for f in os.listdir(words_directory) if f.endswith('.png')]):
            extract_words(os.path.join(words_directory, word), letters_directory, 0)
            resize(letters_directory, letters_directory, 28, 2)
            for letter in sorted([f for f in os.listdir(letters_directory) if f.endswith('.png')]):
                # viewImage(os.path.join(letters_directory, letter), zoom_factor=20) # view character (before it goes into model)
                
                image = np.expand_dims(np.expand_dims(1 - (cv2.imread(os.path.join(letters_directory, letter), cv2.IMREAD_GRAYSCALE) / 255), axis=-1), axis=0) # load image, normalize, invert, prime for insertion to CNN

                prediction = model.predict(image) # make prediction
                predicted_label = np.argmax(prediction, axis=1)[0] # get the most likely label (according to the model)
                predicted_character = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[predicted_label] # map label to a specific character
                # print(f"Predicted character: {predicted_character}")

                cv2.imwrite('ex.png', 255 - cv2.imread(os.path.join(letters_directory, letter), cv2.IMREAD_GRAYSCALE)) # view character (as it goes into the model)
                # viewImage('ex.png', zoom_factor=10)
                outputstring += predicted_character.lower()
            outputstring += ' '
        print(f'{outputstring} - {calculateAccuracy(outputstring)}')
        i = input()
        
    delete_all_files(words_directory)
    delete_all_files(letters_directory)
    os.system('rm ex.png')
    os.system('clear')