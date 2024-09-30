'''activation function and step functions and sigmoid functions
and rectified linear and elu and leaky relu and prelu and softplus'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0) #used to keep the random sample consistent in all subsequent np arrays

'''X = [[1, 2, 3, 2.5], 
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

inputs = [0 , 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

an example of a rectified linear function

for i in inputs:
    #out.append(max(0, i)) an even simpler version
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

#print(output)
c321n.github.io/nueral-network-case-study/ '''

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) #radius
        t = np.linspace(class_number*4, (class_number+1)+4, points)\
            + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X , y = create_data(100, 3)


class layer_dense:
    def __init__(self, n_inputs, n_nuerons): #init = initialize
        self.weights = np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons)) #pass the shape as a paramenter
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

'''rectified linear unit object - one of many different activation functions'''

class activation_relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

'''we will change the shape of the data to match the new 
create_data set that has 2 features in each set so (4,5)
becomes (2,5)'''
layer1 = layer_dense(2,5) 

activation1 = activation_relu()

layer1.forward(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)

'''activation functions can be step functions 
    1, x>0
y={  y is 1 when x>0 and 0 when x<=0
    0, x<=0
this works fine for examples but quickly hits its limit as 
you begin to try to fit more complexx shapes

sigmoid allows for a more granular adjustment but has limitations
y = 1/(1+e)^-x
this helps in the optimizer becausewe can more closely measure
the impact of the adjustments in the formula...contains the vanishing
gradient limitation

rectified liniar is simply
      x, x>0
y = {
      0, x<=0
the most popular activation function in a NN because it is nonlinear
and allows it to clip the function
effectively the weight is the slope of the function and the bias
is the b or offset at the point it activates along the x axis
(a negative weight flips when the activation deactivates)
adding a second layer that bias has the affect of shifting the function
along the y axis and a negative weight creates an upper and lower bound

the idea is to setup pieces of a nonliner function that activates
one after another plotting a small pieces of the function you are trying
to fit to using hte bias to offset and the weights for the slope
with multiple hidden layers taking affect of the output of the previous
layer'''

