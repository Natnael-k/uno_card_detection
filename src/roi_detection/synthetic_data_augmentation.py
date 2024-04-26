import os
import cv2
import numpy as np
import random
import csv


# Function to perform alpha blending and paste the card onto the background image at the specified location
def paste_card(bg_image, card_image, x, y):
    bg_height, bg_width = bg_image.shape[:2]
    card_height, card_width = card_image.shape[:2]
    
    # Ensure the card fits within the background image
    if x + card_width > bg_width or y + card_height > bg_height:
        return None
    
    # Create a copy of the background image
    result = bg_image.copy()
    
    # Define the region of interest where the card will be pasted
    roi = result[y:y+card_height, x:x+card_width]
    
    if len(card_image.shape) == 3:  # If card_image doesn't have an alpha channel
        # Paste the card directly onto the background
        roi[:, :] = card_image[:, :]
    else:  # If card_image has an alpha channel
        # Apply alpha blending to paste the card onto the background
        alpha_card = card_image[:, :, 3] / 255.0
        alpha_bg = 1.0 - alpha_card
        for c in range(0, 3):
            roi[:, :, c] = (alpha_card * card_image[:, :, c] +
                            alpha_bg * roi[:, :, c])
        
    ## Return roi coordinates for training    
    roi_coordinate = [x, y, card_height, card_width]
    
    return result, roi_coordinate

# Function to create annotations for the region of interest (ROI)
def create_annotations(image, roi):
    # Draw bounding box on the image
    x_min, y_min, x_max, y_max = roi
    annotated_image = image.copy()
    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw green bounding box
    
    return annotated_image

# Function to randomly select a location to paste the card onto the background image
def get_random_location(bg_width, bg_height, card_width, card_height):
    x = random.randint(0, bg_width - card_width)
    y = random.randint(0, bg_height - card_height)
    return x, y

# Path to the folder containing original images with cards categorized by color
original_images_folder =  os.path.abspath('../../data/preprocessed_images')

# Path to the folder containing background images
background_images_folder =  os.path.abspath('../../data/background_images')
# Output folder for saving the enhanced dataset
output_image_path =  os.path.abspath('../../data/background_augmented_images')

# CSV file to save the ROI
csv_filename = 'roi_coordinates.csv'
with open(csv_filename, mode ='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['blended_image_new', 'ROI X', 'ROI Y', 'ROI_Z', 'ROI_WIdth', 'ROI_Height'])

# Iterate through each original image
for color_folder in os.listdir(original_images_folder):
    color_folder_path = os.path.join(original_images_folder, color_folder)
    print(color_folder)
    print(color_folder_path)
    print(os.path.isdir(color_folder_path))
    if os.path.isdir(color_folder_path):
        # Iterate through each original image within the color folder
        for original_image_name in os.listdir(color_folder_path):
            if original_image_name.endswith('.PNG') or original_image_name.endswith('.jpg'):
                original_image_path = os.path.join(color_folder_path, original_image_name)
                
                # Load the original image which is croped after preprocessing
                original_image = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
                
                # Iterate through each background image
                for background_image_name in os.listdir(background_images_folder):
                    if background_image_name.endswith('.png') or background_image_name.endswith('.jpeg'):
                        background_image_path = os.path.join(background_images_folder, background_image_name)
                        
                        # Load the background image
                        background_image = cv2.imread(background_image_path, cv2.IMREAD_UNCHANGED)
                        
                        bg_height, bg_width = background_image.shape[:2]
                        
                        #resize the image to fit insdie the background
                        new_width = int(bg_width / 4)
                        new_height = int(bg_height / 4)
                        
                        resized_original_image = cv2.resize(original_image, (new_width, new_height))
                
                        # Randomly select a location to paste the card onto the background image
                        x, y = get_random_location(bg_width, bg_height, resized_original_image.shape[1], resized_original_image.shape[0])
                        
                        # Paste the card onto the background image
                        result_image, roi_coordinate = paste_card(background_image, resized_original_image, x, y)
                        
                        # Save the resulting image
                        if result_image is not None:
                            output_image_name = f'{color_folder}_{original_image_name[:-4]}_on_{background_image_name[:-4]}.png'
                            output_image_fullpath = os.path.join(output_image_path, output_image_name)
                            cv2.imwrite(output_image_fullpath, result_image)
                            
                            # Create annotations for the region of interest -numbers on the card from the top left corner 
                            roi_top_left_corner = (x, y + 50, x + 100, y + 100)  # Format: (x_min, y_min, x_max, y_max)
                            annotated_image = create_annotations(result_image, roi_top_left_corner)
                            
                            # Save annotated image
                            annotated_image_path = os.path.join(output_image_path, f'{output_image_name[:-4]}_annotated.png')
                            cv2.imwrite(annotated_image_path, annotated_image)
                            
                            # Save ROI coordinates to CSV file
                            with open(csv_filename, mode='a', newline='') as csv_file:
                                writer = csv.writer(csv_file)
                                writer.writerow([color_folder, roi_coordinate[0], roi_coordinate[1], roi_coordinate[2], roi_coordinate[3]])
                            
                            print("Comleted")
                            
                          
