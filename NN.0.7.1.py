'''building on 0.5 with the categorical class entropy loss function
to keep with convention I need to capitalize my Classes'''

import numpy as np 

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

X, y = create_data(100, 3) #100 sets of (X, y) values for 3 different classes


class Layer_dense:
    def __init__(self, n_inputs, n_nuerons): #init = initialize weights and biases
        self.weights = np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons)) #pass the shape as a paramenter

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activtion_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis =1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis =1, keepdims=True)
        self.output = probabilities

'''categorical class entropy loss function - how wrong are we?'''
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        batch_loss = np.mean(sample_losses)
        return batch_loss

class Loss_catcross_ent(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: # they have passed scaler values   
            correct_confidence = y_pred_clip[range(samples), y_true]
        elif len(y_true.shape) == 2: # one-hot encoded
            correct_confidence = np.sum(y_pred_clip*y_true, axis =1)
        
        neg_log_likely = -np.log(correct_confidence)
        return neg_log_likely


dense1 = Layer_dense(2,4) #2 here is referring to (X, y) in the set there can be more but we are only working with 2
activation1 = Activation_relu()

dense1.forward(X)
activation1.forward(dense1.output)

dense2 = Layer_dense(4,3) #the output needs to >= the y value at the beginning for the loss calculation
activation2 = Activtion_softmax()

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_funct = Loss_catcross_ent()
loss_ = loss_funct.calculate(activation2.output, y)
print("loss: ", loss_)