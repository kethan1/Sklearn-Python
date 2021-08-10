import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

random_points = [(random.randint(1, 20), random.randint(1, 20)) for _ in range(200)]
print(random_points)

clf = KMeans(2)
clf.fit(random_points)

centroids = clf.cluster_centers_
labels = clf.labels_
print(centroids, labels, random_points)

x = [p[0] for p in random_points]
y = [p[1] for p in random_points]

colors = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'yellow',
    4: 'purple',
    5: 'orange',
    6: 'pink',
    7: 'black',
}

# goes through each points and plots it with the color of the cluster it belongs to
for x_, y_, label in zip(x, y, labels):
    plt.scatter(x_, y_, c=colors[label])

# Plots all the centroids
for centroid in centroids:
    plt.scatter(
        [centroid[0]], [centroid[1]], c="yellow", linewidths=2, marker="^",
        edgecolor="red", s=200
    )

plt.show()
