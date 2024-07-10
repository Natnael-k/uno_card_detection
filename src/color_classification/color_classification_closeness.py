
class ColorRecognizer:
    def __init__(self):
        self.colours = {}

    def estimate_hsv_values(self, images, class_color):
        samples = len(images)
        color = np.zeros((samples, 4))
        colour_dict = {
            "black": 0,
            "blue": 1,
            "green": 2,
            "red": 3,
            "yellow": 4
        }

        for i, image_path in enumerate(images):
            image = cv2.imread(image_path)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            img_ero = cv2.erode(img_th, kernel)

            contours, _ = cv2.findContours(img_ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue  # Skip this image if no contours are found

            largest_contour = max(contours, key=cv2.contourArea)
            imHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            HSV = []
            for point in largest_contour:
                pHSV = imHSV[point[0][1], point[0][0]]
                if pHSV[0] != 0:
                    HSV.append(pHSV)

            if not HSV:
                continue  # Skip this image if no valid HSV values are found

            HSV = np.array(HSV)
            HSVmu = np.mean(HSV, axis=0)
            row_data = np.hstack((HSVmu, colour_dict[class_color]))
            color[i] = row_data

        valid_color_data = color[color[:, 0] != 0]  # Filter out rows with no valid data

        return valid_color_data

    def recognize_color_with_closeness(self, image, colors_feature):
         # Means and standard deviations of H, S, V values learnt by running "colour_estimation" for each object type
        #---> This is the data structure that will be used as color feature, Use this values if needed manual HSV masking values for each color
        # {
        #     'black': [[35.57, 37.35, 193.42], [7.63, 9.79, 13.81]],
        #     'red': [[13.16, 127.28, 177.41], [9.83, 18.00, 32.83]],
        #     'yellow': [[31.09, 101.74, 208.73], [0.94, 15.65, 10.60]],
        #     'blue': [[99.87, 109.19, 220.15], [0.60, 9.25, 20.20]],
        #     'green': [[72.79, 105.61, 155.12], [1.34, 9.95, 15.63]],
        # }
        self.colours = colors_feature
        
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        img_ero = cv2.erode(img_th, kernel)

        contours, _ = cv2.findContours(img_ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None  # No contours found

        largest_contour = max(contours, key=cv2.contourArea)
        imHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        HSV = []
        for point in largest_contour:
            pHSV = imHSV[point[0][1], point[0][0]]
            if pHSV[0] != 0:
                HSV.append(pHSV)

        if not HSV:
            return None, None  # No valid HSV values found

        HSV = np.array(HSV)
        HSVmu = np.mean(HSV, axis=0)

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
