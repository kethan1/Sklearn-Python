import os
import cv2
import numpy as np
from sklearn import neighbors
import sklearn.model_selection

datafolder = "data/knn_image_classifier"
characters = [os.path.join(datafolder, "apples"), os.path.join(datafolder, "donald_duck")]


images = {
    os.path.basename(os.path.normpath(character)): os.listdir(character)
    for character in characters
}


grayscale_images = {
    character: [cv2.imread(os.path.join(datafolder, character, image), cv2.IMREAD_GRAYSCALE) for image in paths]
    for character, paths in images.items()
}


labels = np.asarray([each_character for character, image_lst in grayscale_images.items() for each_character in len(image_lst) * [character]])
data = np.asarray([image for character, image_lst in grayscale_images.items() for image in image_lst])
# print(labels)
# print(data)

nsamples, nx, ny = data.shape
reshaped_data = data.reshape((nsamples, nx * ny))
# cv2.imshow("hi", data)
# cv2.waitKey(0)
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(reshaped_data, labels, test_size=0.001)

clf = neighbors.KNeighborsClassifier()
# Train the classifier
clf.fit(reshaped_data, labels)

# Get the accuracy of the classifier basing on the test data
# accuracy = clf.score(x_test, y_test)
# print(accuracy)

test_image = np.asarray([cv2.imread(os.path.join(datafolder, "apple.jpg"), cv2.IMREAD_GRAYSCALE)])
nsamples, nx, ny = test_image.shape
test_image_reshaped = test_image.reshape((nsamples, nx * ny))
print(clf.predict(test_image_reshaped))

test_image = np.asarray([cv2.imread(os.path.join(datafolder, "donald_duck.jpg"), cv2.IMREAD_GRAYSCALE)])
nsamples, nx, ny = test_image.shape
test_image_reshaped = test_image.reshape((nsamples, nx * ny))
print(clf.predict(test_image_reshaped))
