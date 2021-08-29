import numpy as np
from sklearn import neighbors
import sklearn.model_selection
import pandas as pd


df = pd.read_csv("data/breastcancer/wdbc.csv", comment='#')
df.replace("?", -99999, inplace=True)

y = np.asarray(df["diagnose"])
x = np.asarray(df.drop(["id_number", "diagnose"], 1))

# Test size here is 20% of the data
# X_train, Y_train are the features and labels for the training data
# x_test, y_test are the features and labels for the testing data
X_train, x_test, Y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
# Train the classifier
clf.fit(X_train, Y_train)

# Get the accuracy of the classifier basing on the test data
accuracy = clf.score(x_test, y_test)

print(accuracy)

data = np.asarray([
    17.99, 10.38, 123, 1001, 0.1184, 0.276, 0.3001, 0.1471,
    0.2419, 0.07871, 1, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373,
    0.01587, 0.03003, 0.0041, 25.98, 17.33, 184.6, 2019, 0.1622, 0.6656,
    0.79, 0.2654, 0.4601, 0.12
]).reshape(1, 30)
print(clf.predict(data))
