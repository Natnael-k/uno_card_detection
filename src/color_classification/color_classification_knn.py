import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import cv2

class ColorKNN:
    def recognise_color_with_KNN(self, colours):
        training_set = np.concatenate(colours, axis=0)
        X_train = training_set[:, :-1]  # Extract all rows and all columns except the last one
        y_train = training_set[:, -1]   # Extract all rows and only the last column

        # Train a KNN classifier
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(X_train, y_train)

        # Save the trained model
        with open('knn_model.pkl', 'wb') as file:
            pickle.dump(knn_classifier, file)

    def predict_color(self, image):
        # Load the trained KNN model
        with open('knn_model.pkl', 'rb') as file:
            knn_classifier = pickle.load(file)

        # Process the image and extract HSV features
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        img_ero = cv2.erode(img_th, kernel)

        contours, _ = cv2.findContours(img_ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None  # No contours found

        largest_contour = max(contours, key=cv2.contourArea)
        imHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        HSV = []
        for point in largest_contour:
            pHSV = imHSV[point[0][1], point[0][0]]
            if pHSV[0] != 0:
                HSV.append(pHSV)

        if not HSV:
            return None  # No valid HSV values found

        HSV = np.array(HSV)
        HSVmu = np.mean(HSV, axis=0)

        # Predict the color
        prediction = knn_classifier.predict([HSVmu])
        
        return prediction[0]
