import cv2
import os
import glob
import numpy as np
from color_classification.color_classification_training import ColorDetection
from number_classification.number_classification_training import FeatureDetection
from original_image_preprocessing import ImagePreprocessor

class UnoDetector:
    def __init__(self, model_file='knn_model.pkl'):
        self.original_images_path = os.path.abspath('../data/original_images')
        self.preprocessed_images_path = os.path.abspath('../data/preprocessed_images')
        # Initialize color detection model
        self.color_detector = ColorDetection()
        self.feature_detector = FeatureDetection()
        self.imageprocessing = ImagePreprocessor(self.original_images_path, self.preprocessed_images_path )
        print("Main script intialised")
        
    def train_features_and_color_detectors(self):
        
        #prpepare Data extract features
        self.HSVs_for_kNN_classifier = []
        self.HSVs_for_clossnesses_classifier= []
        self.HSVs_for_mask_classifier = {}
        self.HSVs_for_Segmentation_min = []
        self.HSVs_for_Segmentation_max = []
        self.HSVs_for_clossnesses_classifier_formated = {}
        
        for folder in os.listdir(os.path.abspath("../data/preprocessed_images")):
                                 
            images = glob.glob(os.path.join(os.path.join(os.path.abspath("../data/preprocessed_images"), folder) , "*.PNG"))
            color_mean, color_std, color_min, color_max, color = self.color_detector.estimate_hsv_values(images, folder) 
            self.HSVs_for_Segmentation_min.append(color_min)
            self.HSVs_for_Segmentation_max.append(color_max)
            self.HSVs_for_clossnesses_classifier.append([color_mean, color_std])
            self.HSVs_for_kNN_classifier.append(color)
            self.HSVs_for_mask_classifier[folder] = [[color_min, color_max]]
         
        print('hsv_mask', self.HSVs_for_mask_classifier)    
        #Train color detection Model
        self.color_detector.recognise_color_with_KNN(self.HSVs_for_kNN_classifier) 
        
        #Train Feature Detection Model
        feature_space, lables = self.feature_detector.create_dataset()
        self.feature_detector.train_model(feature_space,lables)
        
        print("Trained feature and color detection models sucessfully")

        
    def detect_uno_cards(self, image): 
        
        # {
        #   by usin hsv trackbar
        #     'yellow': [[22, 145, 47], [37, 252, 144]],
        #     'red': [[38, 77, 57], [179, 255, 237]],
        #     'blue': [[50, 51, 86], [127, 255, 246]],
        #     'green': [[39, 43, 31], [113, 255, 217]],
        # }  
        
        imgContour = image.copy()
        
        hsv_min_list = [[22, 145, 47],[38, 77, 57],[50, 51, 86],[39, 43, 31]] 
        hsv_max_list = [[37, 252, 144],[179, 255, 237],[127, 255, 246],[113, 255, 217]]
        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Iterate through each HSV range and perform segmentation
        for hsv_min, hsv_max in zip(hsv_min_list, hsv_max_list):
            # Create mask for the current HSV range
            lower_bound = np.array(hsv_min, dtype=np.uint8)
            upper_bound = np.array(hsv_max, dtype=np.uint8)
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            cv2.imshow('Mask', mask)
            
            # Apply the mask to the original frame
            segmented_image = cv2.bitwise_and(image, image, mask=mask)
            
            
            #cv2.imshow('boungingbox', segmented_image)
            
    
        
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #largest_contour = max(contours, key=cv2.contourArea)
            #cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
            cv2.imshow('imgContour', imgContour)
            if len(contours) > 0:
                all_contours = np.concatenate(contours)
                x, y, w, h = cv2.boundingRect(all_contours)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                aspect_ratio = w / float(h)
                if aspect_ratio > 0.5 or aspect_ratio < 1.8:  # Adjust range as needed
                    cropped_frame = mask[y:y+h, x:x+w]
                    cv2.rectangle(imgContour, (x,y), (x + w, y + h), (0, 240, 0), 5)
                    
                    
                # cv2.putText(imgContour, "Color_closenes:" + color_results_with_clossness, (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
                # cv2.putText(imgContour, "Color_kNN:" + color_results_knn, (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
                # cv2.putText(imgContour, "Card Number:" + str(shape_results), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
            
                cv2.imshow('cropped_frame', cropped_frame)
            
                shape_results, probalities = self.feature_detector.predict_shape(cropped_frame)
                #print(shape_results)
        #         _, color_results_with_clossness = self.color_detector.recognize_color_with_closeness(cropped_frame, self.HSVs_for_clossnesses_classifier_formated)
        #         print(color_results_with_clossness)
        #         color_results_knn = self.color_detector.predict(cropped_frame)
        #         print(cropped_frame)
        #         peri = cv2.arcLength(contour, True)
        #         approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        #         x, y, w, h = cv2.boundingRect(approx)
        #         
        #         cv2.rectangle(imgContour, (x,y), (x + w, y + h), (0, 240, 0), 5)
        #         cv2.putText(imgContour, "Color_closenes:" + color_results_with_clossness, (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
        #         cv2.putText(imgContour, "Color_kNN:" + color_results_knn, (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
        #         cv2.putText(imgContour, "Card Number:" + str(shape_results), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
            
       
                    
if __name__ == "__main__":
    Uno = UnoDetector()
    #train detectors
    #Uno.train_features_and_color_detectors()
    #predict uno Card
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.resize(frame, (360, 240))
        if not ret:
            break
        Uno.detect_uno_cards(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
   
    print("See you Soon!")
    cap.release()
    cv2.destroyAllWindows()
    
