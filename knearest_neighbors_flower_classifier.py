import numpy as np
from sklearn import preprocessing, neighbors
import sklearn.model_selection 
import pandas as pd


df = pd.read_csv("data/iris/iris.csv", comment='#')
df.replace("?", -99999, inplace=True)

y = np.asarray(df["names"])

x = np.asarray(df.drop(["names"], 1))

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

data = np.asarray([5.5, 3.8, 1.5, 0.1]).reshape(1, 4)
print(clf.predict(data))
