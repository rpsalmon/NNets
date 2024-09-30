''''arrays and shape, you want to keep the shape consistent
list of lists (lol) 
list = [1,2,3,4] has a shape of (1,4)
lol = [[1,2,3,4],
        [5,6,7,8]] has a shape of (2,1,4)
lolol = [[[1,2,3,4],
        [5,6,7,8]],
        [[9,0,1,2],
        [3,4,5,6]]] has a shape of (2,2,4) 
                    2 lists of 2 lists of 4 values
                    a 2d array
lololol = [[[1,2,3,4],
        [5,6,7,8]],
        [[9,0,1,2],
        [3,4,5,6]],
        [[7,8,9,0],
        [1,2,3,4]]] would be a (3,2,4) a 3d array'''

import numpy as np
inputs = [1, 2, 3, 2.5] #vector
weight = [0.2, 0.8, -0.5, 1.0]
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
bias = 2
biases = [2, 3, 0.5]
output = np.dot(weight,inputs)+bias
'''np.dot() will perform multiplication and addition of 
the values in the array by their position like we had 
manually written in NN.0.1 dot will index to the first
variable passed to it in this example weights is a (3,4) 
2d array (or matrix)'''
#print(output)
outputs = np.dot(weights,inputs)+biases
print(outputs)