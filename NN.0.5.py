'''softmax activation used for the output layer  what to do
with negative values in anticipation of back propogatio...
to solve for this we can use euler's number "e" then normalize
and it becomes an output'''
layer_outputs = [4.8, 1.21, 2.385]
import math 
#E = 2.71828182846
E = math.e

exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

#print(exp_values) #exponentiated values

''''normalization into a probability distribution
eliminating the negative values without losing the meaning'''

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

'''simplifying the above using numpy'''
import numpy as np

exp_values = np.exp(layer_outputs) #expontential

norm_values = exp_values / np.sum(exp_values) #normalize

#print(norm_values) #normalized exponentiated values
#print(sum(norm_values)) #validates they sum to 1 

'''the exponentiation or the inputs, normalization makes the 
softmax activation function which creates the output'''

'''working all of this to work with batches'''

layer_batches = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_batch = np.exp(layer_batches)
#print(exp_batch)

norm_batches = exp_batch / np.sum(layer_batches, axis=1,  keepdims=True)
'''sum axis none is a simple sum, 0 will sum columns, 1 will
sum rows, keepdims will align it for '''
#print(norm_batches)

'''because expontentiation grows very quickly, we subtract the 
max value from the output layer from all values effectively
shifting the range to the left of 0 and resulting in an output 
between 0 and 1'''

exp_batch1 = np.exp(layer_batches - np.max(layer_batches, axis =1, keepdims=True))

'''now we add the latest step to the code from previous and def the class 
activation_softmax'''

np.random.seed(0)

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

X, y = create_data(100, 3)


class layer_dense:
    def __init__(self, n_inputs, n_nuerons): #init = initialize
        self.weights = np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons)) #pass the shape as a paramenter
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class activation_relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

'''softmax function'''
class activtion_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis =1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis =1, keepdims=True)
        self.output = probabilities


dense1 = layer_dense(2,3)
activation1 = activation_relu()

dense1.forward(X)
activation1.forward(dense1.output)

dense2 = layer_dense(3,3)
activation2 = activtion_softmax()

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])