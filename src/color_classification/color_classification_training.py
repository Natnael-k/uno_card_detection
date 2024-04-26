import cv2
import numpy as np
import os

# Function to process each image
def process_image(image, label):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold using Otsu's thresholding method
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Erode the thresholded image
    kernel = np.ones((5,5), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    
    # Find contours in the eroded image
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over contours
    for contour in contours:
        # Extract contour points
        for point in contour:
            # Get HSV values of the current pixel
            hsv_pixel = cv2.cvtColor(np.uint8([[image[point[0][1], point[0][0]]]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Filter out contour points with zero hue values
            if hsv_pixel[0] != 0:
                # Append HSV values to the color array
                color_array.append(hsv_pixel)
                # Append label to the labels array
                labels_array.append(label)

# Define folder containing color folders
color_folder_path = 'path/to/your/color/folder'

# Initialize arrays to store HSV values and labels
color_array = []
labels_array = []

# Define labels corresponding to each color folder
color_labels = {'blue': 0, 'red': 1, 'green': 2, 'yellow': 3, 'black': 4}

# Iterate over each color folder
for color_folder in os.listdir(color_folder_path):
    # Construct full path to color folder
    color_folder_fullpath = os.path.join(color_folder_path, color_folder)
    
    # Check if it's a directory
    if os.path.isdir(color_folder_fullpath):
        # Get the label for the current color folder
        label = color_labels[color_folder]
        
        # Iterate over images in the color folder
        for image_name in os.listdir(color_folder_fullpath):
            # Construct full path to image
            image_path = os.path.join(color_folder_fullpath, image_name)
            
            # Read the image
            image = cv2.imread(image_path)
            
            # Process the image
            process_image(image, label)

# Convert color array and labels array to numpy arrays
color_array = np.array(color_array)
labels_array = np.array(labels_array)

# Print the shape of the arrays to verify
print("Color array shape:", color_array.shape)
print("Labels array shape:", labels_array.shape)
