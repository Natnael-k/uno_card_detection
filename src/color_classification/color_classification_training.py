import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier

class ColorDetection:
    
    def __init__(self):
        self.model_file = None
        pass

    def segment_by_color(self, image, HSVs_for_Segmentation):
        hsv_min, hsv_max = HSVs_for_Segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, hsv_min, hsv_max)
        segmented_image = cv2.bitwise_and(image, image, mask=mask)
        
        #also return bbox for segmentation
        return segmented_image

    def estimate_hsv_values(self, images, class_color):
        
        samples = len(images)
        color = np.zeros((samples, 4))
        colour_dict = {
            "black" : 0,
            "blue" : 1,
            "green" : 2,
            "red" : 3,
            "yellow" : 4
        }

        for i in range(len(images)):
            
            image =cv2.imread(images[i]) 
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            img_ero = cv2.erode(img_th, kernel)

            internal_chain, hierarchy = cv2.findContours(img_ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            chain_ar = internal_chain[0]

            imHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            HSV = np.zeros((1, 3))

            for c in range(chain_ar.shape[0]):
                pHSV = imHSV[chain_ar[c][0][1]][chain_ar[c][0][0]]
                if pHSV[0] != 0:
                    HSV = np.vstack((HSV, pHSV))
            HSV = HSV[1:]  

            HSVmu = np.mean(HSV, axis=0)
            row_data = np.hstack((HSVmu, colour_dict[class_color]))
            color[i] = row_data
            
            
           
        print(color)    
        color_mean = np.mean(color[:,:-1], axis=0)
        color_std = np.std(color[:,:-1], axis=0)
        color_min = np.min(color[:,:-1], axis=0)
        color_max = np.max(color[:,:-1], axis=0)
        

        return color_mean, color_std, color_min, color_max, color

    def recognize_color_with_closeness(self, image, colors_feature):
        # Means and standard deviations of H, S, V values learnt by running "colour_estimation" for each object type
        self.colours = colors_feature
        
        # {
        #     'black': [[35.57, 37.35, 193.42], [7.63, 9.79, 13.81]],
        #     'red': [[13.16, 127.28, 177.41], [9.83, 18.00, 32.83]],
        #     'yellow': [[31.09, 101.74, 208.73], [0.94, 15.65, 10.60]],
        #     'blue': [[99.87, 109.19, 220.15], [0.60, 9.25, 20.20]],
        #     'green': [[72.79, 105.61, 155.12], [1.34, 9.95, 15.63]],
        # }

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        img_ero = cv2.erode(img_th, kernel)

        internal_chain, image = cv2.findContours(img_ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        chain_ar = internal_chain[0]
        imHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        HSV = np.zeros(3)
        for c in range(chain_ar.shape[0]):
            pHSV = imHSV[chain_ar[c][0][1]][chain_ar[c][0][0]]
            if pHSV[0] != 0:
                HSV = np.vstack((HSV, pHSV))
        HSV = HSV[1:]

        HSVmu = np.mean(HSV, 0)
        # Closeness is akin to the inverse of probability, lower values correspond to higher probabilities
        closeness = {}
        # Summing the distances of H, S, V from the averages found for each colour, divided by the standard deviations
        for col in self.colours.keys():
            closeness[col] = (abs(HSVmu[0] - self.colours[col][0][0]) / self.colours[col][1][0]) \
                             + abs(HSVmu[1] - self.colours[col][0][1]) / self.colours[col][1][1] \
                             + abs(HSVmu[2] - self.colours[col][0][2]) / self.colours[col][1][2]  # V

        # Dictionary sorted by values
        sorted_colors = sorted(closeness.items(), key=lambda x: x[1])
        
        recognised_card_color = sorted_colors[0] 
        
        return sorted_colors, recognised_card_color
    
    def recognise_color_with_KNN(self, colours):
        
        X1, X2, X3, X4, X5 = colours
        training_set = np.concatenate((X1, X2, X3, X4, X5), axis=0)
        X_train = training_set[:, :-1]  # Extract all rows and all columns except the last one
        y_train = training_set[:, -1]   # Extract all rows and only the last column
        

        # Step 2: Convert the NumPy array to a pandas DataFrame
        column_names = ['Feature1', 'Feature2', 'Feature3']  # Replace with appropriate column names
        df = pd.DataFrame(X_train, columns=column_names)
        df['Label'] = y_train

        # Step 3: Train a KNN classifier
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(X_train, y_train)

        # Save the trained model
        with open('knn_model.pkl', 'wb') as file:
            pickle.dump(knn_classifier, file)


    def predict(self, X_test):
          
        with open(self.model_file, 'rb') as file:
            self.loaded_model = pickle.load(file)
            
        if self.loaded_model is None:
            raise FileNotFoundError("Model file does not exist.")

        # Make predictions using the loaded model
        predictions = self.loaded_model.predict(X_test)
        return predictions