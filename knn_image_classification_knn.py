import os
import cv2
import numpy as np
from sklearn import neighbors
import sklearn.model_selection

datafolder = "data/knn_image_classifier"
characters = [os.path.join(datafolder, "apples"), os.path.join(datafolder, "donald_duck")]

images = {
    path: [cv2.imread(os.path.join(character, image)) for image in os.listdir(character)]
    for character in characters
    if (path := os.path.basename(os.path.normpath(character)))
}

labels = np.asarray([each_character for character, image_lst in images.items() for each_character in len(image_lst) * [character]])
data = np.asarray([image.flatten() for _, image_lst in images.items() for image in image_lst])

# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(reshaped_data, labels, test_size=0.001)

clf = neighbors.KNeighborsClassifier()
# Train the classifier
clf.fit(data, labels)

# Get the accuracy of the classifier basing on the test data
# accuracy = clf.score(x_test, y_test)
# print(accuracy)

test_image = cv2.imread(os.path.join(datafolder, "apple.jpg"))
test_image = np.asarray([test_image.flatten()])
print(clf.predict(test_image))

test_image = cv2.imread(os.path.join(datafolder, "donald_duck.jpg"))
test_image = np.asarray([test_image.flatten()])
print(clf.predict(test_image))
