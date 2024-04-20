import cv2
import numpy as np
import os
import random


""" This script is used for preprocessing images from an original dataset to augument d
 and crop out only the card image for future processing"""

# Function to process each image
def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding (example using Otsu's thresholding method)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Bounding rectangle around the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Bounding rectangle around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    card = image[y:y+h, x:x+w]

    return gray

# Function to augment brightness and contrast
def augment_brightness_contrast(image, alpha_range=(0.8, 1.2), beta_range=(-20, 20)):
    # Generate random alpha and beta values within the specified ranges
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    
    augmented_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return augmented_image

# Define folder containing images categorized by color label
color_folder_path = os.path.abspath('../data/original_images')
output_folder_path = os.path.abspath('../data/preprocessed_images')  # Output folder to save augmented images

# Iterate over each color folder
for color_folder in os.listdir(color_folder_path):
    # Construct full path to color folder
    color_folder_fullpath = os.path.join(color_folder_path, color_folder)

    # Check if it's a directory
    if os.path.isdir(color_folder_fullpath):
        # Create a folder to store processed images for this color
        output_color_folder_path = os.path.join(output_folder_path, color_folder)
        os.makedirs(output_color_folder_path, exist_ok=True)

        # Iterate over images in the color folder
        for image_name in os.listdir(color_folder_fullpath):
            # Construct full path to image
            image_path = os.path.join(color_folder_fullpath, image_name)

            # Read the image
            image = cv2.imread(image_path)

            # Process the image (crop the card)
            processed_image = process_image(image)

            # Augment brightness and contrast
            augmented_image = augment_brightness_contrast(processed_image, alpha_range=(0.8, 1.2), beta_range=(-20, 20))  # Adjust range as needed

            # Save the augmented image
            augmented_image_path = os.path.join(output_color_folder_path, f"{image_name[:-4]}_augmented.png")
            cv2.imwrite(augmented_image_path, augmented_image)

            print(f"Processed and saved: {augmented_image_path}")
