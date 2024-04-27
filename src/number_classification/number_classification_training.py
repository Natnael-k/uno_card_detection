import cv2
import numpy as np
# classifying simple objects based on a selected set of features
import random
import numpy as np
import cv2
import os
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

class FeatureDetection:
    
    """Feature extraction, and training """
    
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


    def get_contours(self, img, HSV_mask_for_feature_recognition):
        
        cv2.imshow("img", img)
        cv2.waitKey(0) 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        imHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        HSVmin, HSVmax = HSV_mask_for_feature_recognition
        cv2.imshow("hsv", imHSV)
        cv2.waitKey(0) 
        print("hsv_mask", HSV_mask_for_feature_recognition)
        img_mask = cv2.inRange(imHSV, HSVmin, HSVmax)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        img_th = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)  # erode + dilate
        cv2.imshow("mak", img_mask)
        cv2.waitKey(0)
        
        contour_list, hierarchy = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        n_obj = len(contour_list)
        ch = 0
        while ch < n_obj and len(contour_list[ch]) < 10:
            ch += 1
            print(contour_list[ch])
        contour = contour_list[ch]
        
        return contour

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
    
    def create_dataset(self,HSVs_for_mask_classifier, n = 25):
        
        selected_dataset = []
        feature_space = []
        labels = []
        # base_dir = os.path.abspath('../../data/preprocessed_images')
        # Iterate over color folders
        for folder in os.listdir(os.path.abspath("../data/preprocessed_images")):
            color_path = os.path.join(os.path.abspath("../../data/preprocessed_images"), folder)
            if os.path.isdir(color_path):
                # Initialize a dictionary to keep track of selected images for each card type
                selected_images = {card_type: False for card_type in self.card_numbers }
                
                # Iterate over card types
                for card_type in self.card_numbers :
                    # Filter images based on card type and preprocessing status
                    images = glob.glob(os.path.join(color_path, f'*_{card_type}_preprocessed.PNG'))
                    if images:
                        # Randomly select 1 image for the current card type
                        selected_image = random.choice(images)
                        selected_dataset.append([selected_image, folder, card_type])
                        selected_images[card_type] = True
        
                # Check if all card types have been selected
                if all(selected_images.values()):
                    # If all card types are present, break the loop
                    break

        # Ensure at least 25 images are selected
        while len(selected_dataset) < n:
            # Randomly select an image from the selected color folder
            random_color_folder = random.choice(os.listdir(os.path.abspath("../data/preprocessed_images")))
            random_color_path = os.path.join(os.path.abspath("../data/preprocessed_images"), random_color_folder)
            random_image = random.choice(glob.glob(os.path.join(random_color_path, '*.PNG')))
            selected_dataset.append([random_image, random_color_folder, random_image[-19:-17]])
        
       
        for dataset in selected_dataset:
            HSV_Mask = HSVs_for_mask_classifier[dataset[1]]
            image = cv2.imread(dataset[0])
            current_contour = self.get_contours(image, HSV_Mask[0])
            new_feature_values = self.find_features(current_contour,self.feature_list)  # chain = contours(i)
            feature_space.append(new_feature_values)
            labels.append(dataset[2])
            
            
                
        return feature_space, labels
    
    def train_model(self, X_train, y_train):
        
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
        
    def predict_shape(self, image, color):
        
        current_contour = self.get_contours(image, color)
        new_feature_values = self.find_features(current_contour, self.feature_list)
        temp = np.asarray(new_feature_values).reshape(1, -1)
        X_test = self.scaler.transform(temp)
        new_F = self.pca.transform(X_test)
        prob = self.classifier.predict_proba(new_F)
        predicted_shape = self.card_numbers[np.argmax(prob)]
        probabilities = ["%0.6f" % p for p in prob[0]]

        return predicted_shape, probabilities
    
    
if __name__ == "__main__":
    
    image = cv2.imread(os.path.abspath("../../data/preprocessed_images/blue/b_08_preprocessed.PNG"))
    HSVmasking = [ np.asarray([99.87, 109.19, 220.15]),   np.asarray([0.60, 9.25, 20.20])]
    det = FeatureDetection()
    det.get_contours(image, HSVmasking)

# {'black': [[array([ 34.15037966,  14.71207953, 132.46243718]), array([3.77269726, 1.54113936, 5.04183632])]], 
# 'blue': [[array([ 27.68814548,  18.85656918, 174.64061   ]), array([ 0.89614955,  1.56607071, 31.00258479])]], 
# 'green': [[array([ 25.47568036,  18.19218159, 174.60378035]), array([ 0.8956469 ,  2.25711578, 22.57357425])]], 
# 'red': [[array([ 34.07084413,  14.88664041, 145.83160265]), array([ 0.7881771 ,  1.57768108, 10.93406817])]], 
# 'yellow': [[array([ 31.25363117,  13.42204303, 177.73161954]), array([ 2.51840713,  1.48964919, 21.57382226])]]}