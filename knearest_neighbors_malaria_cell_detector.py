import os
import math
import cv2
import numpy as np
from sklearn import neighbors
import sklearn.model_selection

datafolder = "data/malaria_data_set/cell_images"
cell_paths = [os.path.join(datafolder, "Parasitized"), os.path.join(datafolder, "Uninfected")]

images = {}

max_size_x = 0
max_size_y = 0

for cell_path in cell_paths:
    path = os.path.basename(os.path.normpath(cell_path))
    images[path] = []
    for image in os.listdir(cell_path)[:2000]:
        img = cv2.imread(os.path.join(datafolder, path, image))
        max_size_x, max_size_y = max(max_size_x, img.shape[1]), max(max_size_y, img.shape[0])
        images[path].append(img)

for path, image in images.items():
    for index, img in enumerate(image):
        y_padding = (max_size_y - img.shape[0]) / 2
        x_padding = (max_size_x - img.shape[1]) / 2
        images[path][index] = cv2.copyMakeBorder(
            img, math.ceil(y_padding), math.floor(y_padding),
            math.ceil(x_padding), math.floor(x_padding),
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )


labels = np.asarray([each_character for character, image_lst in images.items() for each_character in len(image_lst) * [character]])
data = np.asarray([image.flatten() for image_lst in images.values() for image in image_lst])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
# Train the classifier
clf.fit(x_train, y_train)

# Get the accuracy of the classifier basing on the test data
accuracy = clf.score(x_test, y_test)
print(accuracy)
