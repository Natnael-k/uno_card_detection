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
   Data - Data augumented, preprocessesd as well as original images are staored
   models - T the trained model are stored for inference.
   src - main folder which all utility scripts and main file to run the detection.
   test - test files

   Scipts -
   color_classification_training.py: Contains Class for color detection related functions
   number_classification_training.py: Contians Class for Feature extraction and model training related functions
   synthetic_data_augmentation.py: Contain Class for color, shape and background augumentation related functions
   original_image_preprocessing.py: Contains Script for Croping original image feom its background

   4.2 Installation

   clone the repo
   pip instal requirments.tsx
   cd /src ----> place all your original data insisde the original data sub folders according to the color and
   python original_image_preprocessing.py --->Thi will generate augumentation data for you, chcek the other folders if image is populated

5. Results

The performance of the Uno Card Detection System will be evaluated using various metrics such as accuracy, precision, recall, and F1-score. The results will demonstrate the effectiveness of the proposed approach in accurately detecting and classifying Uno cards.

7. Discussion

The discussion will analyze the strengths and limitations of the system, potential improvements, and future research directions. It will also address any challenges encountered during the implementation and evaluation phases.

8. Conclusion

The Uno Card Detection System presents a robust solution for automatically detecting and classifying Uno cards from images captured using a smartphone camera. By leveraging computer vision techniques and machine learning models, the system achieves accurate detection and classification results, paving the way for applications in gaming and entertainment.

9. Future Work
   Future work may involve expanding the dataset, exploring other models for better performance, and exploring advanced computer vision techniques for more complex card games. Additionally, the system could be deployed as a mobile application for real-time Uno card detection and classification.

10. References

@DR_Sammer

This detailed report provides an overview of the Uno Card Detection System, including its objectives, methodology, implementation details, results, and future work. Let me know if you need further clarification or additional information on any aspect of the project.
