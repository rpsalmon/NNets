'''batches and batch sizes'''
from typing import ForwardRef
import numpy as np

inputs = [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

#outputs = np.dot(weights,inputs)+biases
'''currently has a shape problem and will need to transpose
the values to correct for that...rooted in the way dot product
performs the multiplcation of rows and columns
to do this we need to convert to a numpy array and 
transpose (T)...shape (3,4) needs a shape of (4,3) to dot product'''
outputs = np.dot(inputs,np.array(weights).T)+biases

#print(outputs)

'''additional layers taking the outputs from the first
layer as the input for the second layer(node)'''

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

l1_outputs = np.dot(inputs,np.array(weights).T)+biases

l2_outputs = np.dot(l1_outputs,np.array(weights2).T)+biases2

#print(l2_outputs)

'''change these weights and biases to objects and rename inputs to X
as the training data set'''

np.random.seed(0) #used to keep the random sample consistent in all subsequent np arrays

X = [[1, 2, 3, 2.5], #size of a single batch is 4 or n_inputs
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

'''the size of the inputs (n_inputs) and the weights (n_nuerons) combine to
make the shape...weights creates an array in the shape according to n_inputs and 
n_nuerons with biases returning an array filled with zeros (we start with zeros 
and can adjust this later depending on need...forward prop output with np.dot()
of inputs and weights plus biases as in the earlier function'''

class layer_dense:
    def __init__(self, n_inputs, n_nuerons): #init = initialize
        self.weights = 0.1 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons)) #pass the shape as a paramenter
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

'''np.random.randn produces an vector or array (depending on the shape passed)
that is filled with values of a normal distribution with a mean of 0 and
variance of 1'''
#print(np.random.randn(4,51)) 

'''layer1 equals a layer_dense object and specify the size of the inputs (how 
many features in each sample) and nuerons...for layer2 the input has to match
the output from layer 1...because the np.dot() only cares about the inner values
of two shapes matching the number of nuerons can be any value
how many features in each sample'''

layer1 = layer_dense(4,5) #the input has to match the number of features in each sample in this case 4
layer2 = layer_dense(5,2) #the input has to match the output of layer 1

'''the self call in the layer_dense is literally itself and is skipped when 
passing values...layer1 calls the forward founction and passes the values in X
as inputs to perform the np.dot() operation which was initialized with the shape
of (4,5) where the weights and biases were calculated...layer2 takes the array
produced in layer 1 and because the inner values of each shape match they np.dot()
can be performed again...regardless there will always be 3 samples in any array in
the output because there are 3 samples in the initial set, the number of columns will
change...the inner values of the shapes used in the dot product operation must match
and the output shape will be the outer values'''

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output) #passes the output 4,5 to 5,2 and performs np.dot()
print(layer2.output)