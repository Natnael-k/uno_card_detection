import cv2
import numpy as np
import os


# Function to apply hue shift
def hue_shift(image, delta):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Function to apply saturation adjustment
def saturation_adjustment(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Function to apply brightness and contrast adjustment
def brightness_contrast_adjustment(image, brightness=0, contrast=1):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

# Function to apply geometric transformations
def apply_geometric_transformations(image):
    # Randomly choose transformation parameters
    rows, cols, _ = image.shape
    angle = np.random.uniform(-10, 10)  # Rotation angle range: -10 to 10 degrees
    scale = np.random.uniform(0.8, 1.2)  # Scaling factor range: 0.8 to 1.2
    dx = np.random.randint(-20, 20)  # Translation along x-axis range: -20 to 20 pixels
    dy = np.random.randint(-20, 20)  # Translation along y-axis range: -20 to 20 pixels
    
    # Compute transformation matrix
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # Apply transformations
    transformed_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    transformed_image = cv2.warpAffine(transformed_image, translation_matrix, (cols, rows))
    
    return transformed_image

# Function to augment an image with random parameters
def augment_image(image):
    # Apply geometric transformations
    transformed_image = apply_geometric_transformations(image)
    
    # Randomly choose augmentation parameters for color augmentation
    hue_delta = np.random.randint(-10, 10)  # Hue shift range: -10 to 10 degrees
    saturation_factor = np.random.uniform(0.5, 1.5)  # Saturation adjustment range: 0.5 to 1.5
    brightness = np.random.randint(-50, 50)  # Brightness adjustment range: -50 to 50
    contrast = np.random.uniform(0.8, 1.2)  # Contrast adjustment range: 0.8 to 1.2
    
    # Apply color augmentation
    augmented_image = hue_shift(transformed_image, hue_delta)
    augmented_image = saturation_adjustment(augmented_image, saturation_factor)
    augmented_image = brightness_contrast_adjustment(augmented_image, brightness, contrast)
    
    return augmented_image

# Function to iterate through folders of images and apply augmentation
def augment_images_in_folders(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            output_folder_path = os.path.join(output_dir, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(folder_path, filename)
                    output_path = os.path.join(output_folder_path, filename)
                    augumented_path = os.path.join(output_folder_path,  filename)
                    # Read original image
                    original_image = cv2.imread(image_path)
                    # Apply augmentation
                    augmented_image = augment_image(original_image)
                    # Concatenate original and augmented images horizontally
                    #output_image = np.concatenate((original_image, augmented_image), axis=1)
                    # Save original and augmented images side by side
                    #cv2.imwrite(output_path, original_image)
                    cv2.imwrite(output_path, augmented_image)

# Main function
if __name__ == "__main__":
    input_root_dir = '../data/original_images/'  # Change this to the directory containing folders of images
    output_root_dir = '../data/augmented_images/'  # Change this to the directory where augmented images will be saved
    augment_images_in_folders(input_root_dir, output_root_dir)
    print("Augmentation complete.")
