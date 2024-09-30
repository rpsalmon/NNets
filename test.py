import numpy as np
eye = np.eye(3)[2]
#print(eye)

X = [[1, 2, 3, 2.5], 
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

inputs = [0 , 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

samples = 10

X[range(samples), inputs] -= 1
# normalize the gradient 
X = X / samples