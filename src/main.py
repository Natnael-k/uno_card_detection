import cv2
import os
import glob
from color_classification.color_classification_training import ColorDetection
from number_classification.number_classification_training import FeatureDetection

class UnoDetector:
    def __init__(self, model_file='knn_model.pkl'):
        # Initialize color detection model
        self.color_detector = ColorDetection()
        self.feature_detector = FeatureDetection()
        print("Main script intialised")
        
    def train_features_and_color_detectors(self):
        
        #prpepare Data extract features
        self.HSVs_for_kNN_classifier = []
        self.HSVs_for_clossnesses_classifier= []
        self.HSVs_for_mask_classifier = {}
        self.HSVs_for_Segmentation = []
        self.HSVs_for_clossnesses_classifier_formated = {}
        
        for folder in os.listdir(os.path.abspath("../data/preprocessed_images")):
                                 
            images = glob.glob(os.path.join(os.path.join(os.path.abspath("../data/preprocessed_images"), folder) , "*.PNG"))
            color_mean, color_std, color_min, color_max, color = self.color_detector.estimate_hsv_values(images, folder) 
            self.HSVs_for_Segmentation.append([color_min,color_max])
            self.HSVs_for_clossnesses_classifier.append([color_mean, color_std])
            self.HSVs_for_kNN_classifier.append(color)
            self.HSVs_for_mask_classifier[folder] = [[color_min, color_max]]
         
        print('hsv_mask', self.HSVs_for_mask_classifier)    
        #Train color detection Model
        self.color_detector.recognise_color_with_KNN(self.HSVs_for_kNN_classifier) 
        
        #Train Feature Detection Model
        feature_space, lables = self.feature_detector.create_dataset(self.HSVs_for_mask_classifier)
        self.feature_detector.train_model(feature_space,lables)
        
        print("Trained feature and color detection models sucessfully")
        

    def detect_uno_cards(self, frame):
        
        cap = cv2.VideoCapture(0)
        while True:
            
            ret, frame = cap.read()
            cv2.resize(frame, (360, 240))
            if not ret:
                break
                    
            # Mask the image using segmentation
            masked_image = self.color_detector.segment_by_color(frame, self.HSVs_for_Segmentation)

            # Detect color using color detection
            color_results_knn = self.color_detector.predict(masked_image)
            _, color_results_with_clossness = self.color_detector.recognize_color_with_closeness(masked_image, self.HSVs_for_clossnesses_classifier_formated)
            
            # Detect shape or feature for the second number
            HSV_mask_for_feature_recognition = self.HSVs_for_clossnesses_classifier_formated[color_results_with_clossness][0]
            shape_results = self.feature_detector.predict_shape(masked_image, HSV_mask_for_feature_recognition)

            # Draw bounding box and label the color and card number
            cv2.put

            #cv2.imshow('Frame', frame_undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            
        print("See you Soon!")
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    Uno = UnoDetector()
    #train detectors
    Uno.train_features_and_color_detectors()
    Uno.detect_uno_cards()
    
    
    
