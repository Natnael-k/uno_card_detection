import cv2
import numpy as np
# classifying simple objects based on a selected set of features
import random
import numpy as np
import cv2
import os
import glob
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

class FeatureDetection:
    
    """ Feature extraction, and training """
    
    def __init__(self):
        
        self.curvature_threshold = 0.08
        self.k = 4
        self.polygon_tolerance = 0.04
        self.preprocessed_img_folder = os.path.abspath('../../data/preprocessed_images')
        self.logfile = open('log.txt', 'w')
        self.card_color = ['black','blue','red','green','yellow']
        self.card_numbers = ['00','01','02','03','04','05','06','06','07','08','09','ar', 'dr', 'rev' ]
        
        # features used in object classification
        self.features_dict = ['axes_ratio','concavity_ratio','convexity_ratio','area_ratio','vertex_approx','length','perimeter_ratio']
        self.use_features = [0, 1, 2, 3, 4, 5, 6] # 0..6
        self.feature_list = [self.features_dict[ft] for ft in self.use_features]


    def get_contours(self, img):
        
        #--> Use this for getting contours from the image
        
        # blur the image  
        imgBlur = cv2.GaussianBlur(img, (7,7), 3)
        
        # Convert to grayscale
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        
        #Canny image Detector
        imgCanny = cv2.Canny(imgGray, 97, 87)
        
        #Filter noise by dialtion
        kernel = np.ones((5,5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        
        contour_list, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour_list) > 0:
            n_obj =  len(contour_list)
            ch = 0
            while ch < n_obj and len(contour_list[ch]) < 10:
                ch += 1
                
            contour = contour_list[ch]
            return contour
        else:
            return None
    
    
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

    def extract_training_features(self, contours, features):
        feature_values_list = []
        for contour in contours:
            feature_values = self.find_features(contour, features)
            feature_values_list.append(feature_values)
        return feature_values_list

    def find_features(self, input_contour, features):
        
        curvature_chain = []
        cont_ar = np.asarray(input_contour)

        # compute axes feature
        ellipse = cv2.fitEllipse(cont_ar)
        (center, axes, orientation) = ellipse
        majoraxis_length = max(axes)
        minoraxis_length = min(axes)
        axes_ratio = minoraxis_length / majoraxis_length

        area = cv2.contourArea(cont_ar)
        perimeter = cv2.arcLength(cont_ar, True)
        area_ratio = perimeter / area
        perimeter_ratio = minoraxis_length / perimeter

        epsilon = self.polygon_tolerance * perimeter
        vertex_approx = 1.0 / len(cv2.approxPolyDP(cont_ar, epsilon, True))
        length = len(input_contour)

        # compute curvature and convexity features
        for i in range(cont_ar.shape[0] - self.k):
            num = cont_ar[i][0][1] - cont_ar[i - self.k][0][1]  # y
            den = cont_ar[i][0][0] - cont_ar[i - self.k][0][0]  # x
            angle_prev = -np.arctan2(num, den) / np.pi

            num = cont_ar[i + self.k][0][1] - cont_ar[i][0][1]  # y
            den = cont_ar[i + self.k][0][0] - cont_ar[i][0][0]  # x
            angle_next = -np.arctan2(num, den) / np.pi

            new_curvature = angle_next - angle_prev
            curvature_chain.append(new_curvature)

        convexity = 0
        concavity = 0
        for i in range(len(curvature_chain)):
            if curvature_chain[i] > self.curvature_threshold:
                convexity += 1
            if curvature_chain[i] < -self.curvature_threshold:
                concavity += 1

        convexity_ratio = convexity / float(i + 1)
        concavity_ratio = concavity / float(i + 1)

        feature_values = [eval(ft) for ft in features]
        
        return feature_values
    
    def create_dataset(self):
        
        selected_dataset = []
        feature_space = []
        labels = []
        
        #Iterate over color folders
        for folder in os.listdir(os.path.abspath("../data/preprocessed_images")):
            color_path = os.path.join(os.path.abspath("../data/preprocessed_images"), folder)
            if os.path.isdir(color_path):
                 for image_name in os.listdir(color_path):
                    image_path = os.path.join(color_path, image_name)
                    image = cv2.imread(image_path)
                    current_contour = self.get_contours(image)
                    new_feature_values = self.find_features(current_contour,self.feature_list)
                    print(image_name)# chain = contours(i)
                    feature_space.append(new_feature_values)
                    image_lable = image_name[-19:-17]
                    labels.append(image_lable)
            
            print("feature_space",feature_space)
            print('lables', labels)
                 
        return feature_space, labels
    
    def train_model(self, X_train, y_train):
        
        self.classifier_model_path = os.path.abspath('../number_classification/gaussian_classifier_model.pkl')
        
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        self.pca = PCA(3)  # Put a number < n_feature
        self.pca.fit(X_train_scaled)
        eigenvalues = self.pca.explained_variance_ratio_
        print(eigenvalues)
        X_train_pca = self.pca.transform(X_train_scaled)

        self.classifier = GaussianNB()  # Using Gaussian Naive Bayes classifier
        self.classifier.fit(X_train_pca, y_train)
        
        
        #saving the trainned  model
        with open(self.classifier_model_path, 'wb') as file:
            pickle.dump({
                'scaler': self.scaler,
                'pca': self.pca,
                'classifier': self.classifier
            }, file)
        
    def predict_shape(self, image):
        self.classifier = GaussianNB()
        current_contour = self.get_contours(image)
        new_feature_values = self.find_features(current_contour, self.feature_list)
        temp = np.asarray(new_feature_values).reshape(1, -1)
        X_test = StandardScaler.transform(temp)
        new_F = PCA.transform(X_test)
        prob = self.classifier.predict_proba(new_F)
        predicted_shape = self.card_numbers[np.argmax(prob)]
        probabilities = ["%0.6f" % p for p in prob[0]]

        return predicted_shape, probabilities
    
