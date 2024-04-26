import cv2
import numpy as np
import os
import random



class ImagePreprocessor:
    def __init__(self, original_image_path, preprocessed_image_path):
        """ This script is used for preprocessing images from an original dataset to augument data
        and crop out only the card image for future processing"""
        
        self.original_images_path = original_image_path
        self.preprocessed_images_path = preprocessed_image_path
        
    
        
    def empty(self, a):
        pass

    # Function to process each image
    def preprocess_image(self, image):
        
        #create trackbars
        # cv2.namedWindow("Parameters")
        # cv2.resizeWindow("Parameters", 640, 240)
        # cv2.createTrackbar("Threshold1", "Parameters", 87, 255, self.empty)
        # cv2.createTrackbar("Threshold2", "Parameters", 97, 255, self.empty)
        # cv2.createTrackbar("Area", "Parameters", 0, 1500, self.empty)
      
        #Read the image
        img = image
        
        #copy image to display contour
        imgContour = img.copy()
    
        # blur the image 
        imgBlur = cv2.GaussianBlur(img, (7,7), 3)
        
        # Convert to grayscale
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        
        #save tresholds 
        # threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        # threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        
        #Canny image Detector
        imgCanny = cv2.Canny(imgGray, 97, 87)
        
        #Filter noise by dialtion
        kernel = np.ones((5,5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        
        
        # Find contours
        contours, _ = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        #Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        
        cv2.drawContours(imgContour, largest_contour, -1, (255, 0, 255), 7)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        print(len(approx))
        x_, y_, w_, h_ = cv2.boundingRect(approx)
        cv2.rectangle(imgContour, (x_,y_), (x_ + w_, y_ + h_), (0, 240, 0), 5)
        cv2.putText(imgContour, "Points:" +str(len(approx)), (x_ + w_ + 20, y_ + 20), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
        cv2.putText(imgContour, "Area:" +str(area), (x_ + w_ + 20, y_ + 45), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
    
        #Stack images for visualising
        #imgStack = self.stackImages(0.4, ([img, imgBlur, imgCanny],[imgDil, imgContour, imgContour]))
        
        # Bounding rectangle around the largest contour
        #x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Ensure aspect ratio is within a certain range
        aspect_ratio = w_ / float(h_)
        if aspect_ratio < 0.5 or aspect_ratio > 1.8:  # Adjust range as needed
            return None  # Skip processing if aspect ratio is not within desired range
        
        card = img[y_:y_+h_, x_:x_+w_]

        return card

    # Function to augment brightness and contrast
    def augment_brightness_contrast(self, image, alpha_range=(0.8, 1.2), beta_range=(-20, 20)):
        # Generate random alpha and beta values within the specified ranges
        alpha = random.uniform(*alpha_range)
        beta = random.uniform(*beta_range)
        
        augmented_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return augmented_image


    #src
    # https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/blob/master/Basics/Joining_Multiple_Images_To_Display.py

    # Function to stack images together for visuallisation
    def stackImages(self,scale,imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor= np.hstack(imgArray)
            ver = hor
        return ver

    # Define folder containing images categorized by color label
    def process_all_images(self):
        # Iterate over each color folder
        for color_folder in os.listdir(self.original_images_path):
            # Construct full path to color folder
            color_folder_fullpath = os.path.join(self.original_images_path, color_folder)

            # Check if it's a directory
            if os.path.isdir(color_folder_fullpath):
                # Create a folder to store processed images for this color
                output_color_folder_path = os.path.join(self.preprocessed_images_path, color_folder)
                os.makedirs(output_color_folder_path, exist_ok=True)

                # Iterate over images in the color folder
                for image_name in os.listdir(color_folder_fullpath):
                    # Construct full path to image
                    image_path = os.path.join(color_folder_fullpath, image_name)

                    # Read the image
                    image = cv2.imread(image_path)

                    # Process the image (crop the card)
                    processed_image = self.preprocess_image(image)

                    # Augment brightness and contrast
                    augmented_image = self.augment_brightness_contrast(processed_image, alpha_range=(0.8, 1.2), beta_range=(-20, 20))  # Adjust range as needed

                    # Save the augmented image
                    augmented_image_path = os.path.join(output_color_folder_path, f"{image_name[:-4]}_preprocessed.PNG")
                    cv2.imwrite(augmented_image_path, augmented_image)

                    print(f"Processed and saved: {augmented_image_path}")

def main():
        # Define paths for original and preprocessed images
        original_images_path = os.path.abspath('../data/original_images')
        preprocessed_images_path = os.path.abspath('../data/preprocessed_images')
        test_images_path = os.path.abspath('../data/test/test_image.jpg')

        # Initialize ImagePreprocessor
        image_preprocessor = ImagePreprocessor(original_images_path, preprocessed_images_path)

        # Process all images
        image_preprocessor.process_all_images()
        
        # Preprocess image
        #image_preprocessor.preprocess_image(test_images_path) 
        

if __name__ == "__main__":
    main()