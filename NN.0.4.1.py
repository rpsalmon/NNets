from typing import ChainMap
import numpy as np
import matplotlib.pyplot as plt

'''how many featuresets per classes in this case x and y
2 features that describe what class a particular point is
...in this case we have 3 classes of 100 featureset of 2
features resulting in 300 data points'''

def create_data(points, classes):
    x = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) #radius
        t = np.linspace(class_number*4, (class_number+1)+4, points)\
            + np.random.randn(points)*0.2
        x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return x, y

print("here")
x , y = create_data(100, 3)

plt.scatter(x[:,0], x[:,1])
plt.show()

plt.scatter(x[:,0], x[:,1], c=y, cmap="brg")
plt.show()

