import numpy as np
from sklearn import neighbors
import sklearn.model_selection
import pandas as pd


df = pd.read_csv("data/pokemon.csv", comment='#')
df.replace("?", -99999, inplace=True)
df.fillna(0, inplace=True)

y = np.asarray(df["name"])

df.drop(["japanese_name", "name", "abilities", "classfication", "type1", "type2"], 1, inplace=True)

# df['abilities'] = df['abilities'].apply(pd.to_numeric)
# df['classfication'] = df['classfication'].apply(pd.to_numeric)
# df['type1'] = df['type1'].apply(pd.to_numeric)
# df['type2'] = df['type2'].apply(pd.to_numeric)

x = np.asarray(df)

# Test size here is 20% of the data
# X_train, Y_train are the features and labels for the training data
# x_test, y_test are the features and labels for the testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
# Train the classifier
clf.fit(x_train, y_train)

# Get the accuracy of the classifier basing on the test data
accuracy = clf.score(x_test, y_test)

print(accuracy)


data = np.asarray([1, 1, 1, 2, 1, 1, 0.5, 1, 1, 2, 1, 0.5, 1, 1, 1, 1, 0.5, 0.5, 63, 5120, 70, 405, 45, 80, 1059860, 1.0, 59, 88.1, 8, 65, 80, 58, 22.5, 1, 0]).reshape(1, 35)
print(clf.predict(data))
