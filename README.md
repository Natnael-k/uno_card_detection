Uno Card Detection System using Computer Vision

1. Introduction

-The Uno Card Detection System aims to develop a computer vision solution for detecting and classifying Uno cards from images captured using a smartphone camera. The system will utilize image processing techniques, data augmentation, and machine learning models to accurately detect the cards and classify their attributes such as color and number.

2. Objectives

-Develop a dataset containing images of Uno cards captured from a smartphone camera.
-Implement image processing techniques to preprocess the images and extract the Uno cards.
-Augment the dataset by applying shape transformation and color augmentation to the extracted card images.
-Train a regression convolutional neural network (CNN) model to detect the region of interest (ROI) of the Uno cards.
-Train a GaussianNB() model to classify card numbers based on selected features
-Train a color classifier using a K-nearest neighbors (KNN) model to classify the color of the Uno cards.

3. Methodology

   3.1 Dataset Collection
   The dataset will be collected by capturing and naming images of Uno cards using a smartphone camera using a black background.

   3.2 Image Preprocessing
   Image preprocessing techniques such as filtering, dilation, and morphological operations will be applied to extract the Uno cards from the background. The images will be cropped to isolate individual cards.

   3.3 Data Augmentation
   The cropped card images will undergo shape transformation and color augmentation to increase the diversity of the dataset. Geometric transformations will be applied to simulate variations in card orientation and perspective. Additionally, color augmentation will be performed to simulate variations in lighting conditions and card appearance. Finally, the cropped image will be blended and stiched on top of back ground images that will be used for machine learning based ROI regressions, The data is augumented and generated but the Model Training will be included on future work.

   3.4 ROI Detection
   A regression R-CNN model will be trained to detect the region of interest (ROI) of the Uno cards within the images. The model will predict the bounding box coordinates of each card in the image.The data from the the augumented merged by the backgound and the original image will be used for both labling and anootating the dataset as it is stiched on a random xy coorinate which is extracted as the stiching goes by.

   3.5 Number Classification
   A GaussianNB() model will be trained to classify the numbers and special characters on the Uno cards. The model will learn to recognize the different symbols present on the cards by first extracting features after the image is masked for certain color and the countours are used to calculate aspect-ratio, area, perimeter and more as features.

   3.6 Color Classification
   A K-nearest neighbors (KNN) model will be trained to classify the color of the Uno cards. The model will use the color information in the HSV space- HSVmin, HSVmax, HSV median and HSV standard Deviation - extracted from multiple countors on top of sample cards. Both KNN and closseness calculations are used to classify the color of the card in the project and compared.

4. Implementation

   4.1 Folder Structure
   Data - Data augmented, preprocessed, and original images are staored
models - The trained models are stored for inference.
   src - main folder which all utility scripts and main files to run the detection.
   test - test files

   Scripts -

   color detection ->
   color_classification_closeness.py: The ColorRecognizer class is designed to recognize the color of objects in images based on their HSV values. It estimates the HSV values from a set of images and uses these values to recognize the color of objects in new images through a closeness metric.
   color_classification_knn.py: Contains The ColorKNN class which is designed to recognize colors in images using a K-Nearest Neighbors (KNN) classifier. It includes methods to train the KNN model with provided color data and to predict the color of objects in new images based on their HSV values.

   number classifications ->
   number_classification_training.py: The FeatureDetection class is designed to classify simple objects based on a selected set of features extracted from images. It uses various image processing techniques and machine learning algorithms to achieve this. The class provides methods for feature extraction, dataset creation, model training, and object classification.

   ROI detection -
   synthetic_data_augmentation.py: Contain Class for color, shape, and background augmentation related functionsThis script performs data augmentation by blending card images onto various background images and generating annotations for the regions of interest (ROI). The goal is to create an enhanced dataset for training machine learning models. The script includes several functions and a main loop to automate this process.
   roi_hsv_segmentation: Contains The UnoCardSegmentationDetector class which provides functionality to detect Uno cards based on HSV (Hue, Saturation, Value) color segmentation. It includes methods to segment the cards by their predefined color ranges and draw contours around the detected cards.
   roi_regression_cnn_model: Contains The UnoCardDetector class which provides functionality to detect and localize Uno card regions of interest (ROI) using TensorFlow/Keras for machine learning and OpenCV for image processing.
   synthetic_data_augmentation: The provided script integrates various functionalities using OpenCV and CSV handling to process and augment images, primarily for creating annotated datasets. Here’s a breakdown and review of its components


   Main.py: The UnoDetector class is designed to perform real-time detection, recognition, and annotation of UNO cards using computer vision techniques. Here’s a comprehensive overview of its components and functionality

   utils.py: The ImagePreprocessor class is designed to preprocess images from an original dataset, primarily focusing on extracting and augmenting UNO card images for further processing. Here’s a detailed review of its components and functionality:

   4.2 Installation

   clone the repo
   pip install requirements.txt
   cd /src ----> Place all your original data inside the original data subfolders according to the color and
   python utils.py --->Thi will generate augmentation data for you, check the other folders if the image is populated

5 . Conclusion and Results Analysis

5.1 Results of Color Classification Using KNN and Closeness Models

The Uno Card Detection System achieved promising results in color classification using two distinct models: K-Nearest Neighbors (KNN) and Closeness-based classification. The KNN model demonstrated an accuracy of 87%, effectively distinguishing between Uno card colors under varying conditions. On the other hand, the Closeness model performed exceptionally well in well-lit environments, leveraging subtle color variations to accurately classify Uno card colors.

5.2 Evaluation of ROI Training for Uno Card Detection

The training of Region of Interest (ROI) detection posed significant computational challenges, particularly when executed on a CPU. Despite efforts to train the model using a small image dataset, the achieved accuracy was not satisfactory. This limitation underscores the need for augmenting the dataset with more diverse images and exploring methods to increase its size. Future enhancements should focus on optimizing training parameters and leveraging GPU acceleration to improve computational efficiency.

5.3 Challenges in Number Classification

Number classification emerged as the most challenging aspect of the Uno Card Detection System. This task requires robust data preprocessing due to variations in card orientation, lighting conditions, and image quality. Effective preprocessing techniques are crucial to enhance the accuracy and reliability of number recognition, demanding further research into advanced image processing methodologies.

5.4 Conclusion

In conclusion, the Uno Card Detection System demonstrates promising capabilities in color classification, with the KNN model achieving an accuracy of 87% and the Closeness model excelling under optimal lighting conditions. However, challenges persist in ROI training due to computational constraints and in number classification, which necessitates more sophisticated data preprocessing techniques. Moving forward, expanding the dataset, augmenting data, and refining preprocessing methodologies will be pivotal in advancing the system's overall accuracy and robustness. These efforts will pave the way for broader applications of computer vision in card game analysis and beyond.

References
https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://www.murtazahassan.com/
@DR_Sammer Lecture Notes
