'''Introduction to optimization using a more simpler vertically differentiated dataset'''

import numpy as np

def create_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3)

class Layer_dense:
    def __init__(self, n_inputs, n_nuerons): #init = initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_nuerons)
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

dense1 = Layer_dense(2, 3)
activation1 = Activation_relu()
dense2 = Layer_dense(3, 3)
activation2 = Activtion_softmax()

loss_function = Loss_catcross_ent()

'''some initial lowest loss value will later be set to the loss that is
found with the initial weights and biases'''
lowest_loss = 2 


'''creating new variables for the best weight and bias, initialized as a randomly selected value
using np.radnom.randn...this version will simply test at random different values and printing that loss
and accuracy and saving those weights and biases if the loss is lower than the random one above
the below iss simply storing the initial weights and biases'''
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for i in range(100000):

    #update weights with some small incremental increase to a random value
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    #forward pass the training data for weights biases
    #then for each activation function relu and softmax
    dense1.forward(X) #layer one weights and biases
    activation1.forward(dense1.output) #activation function
    dense2.forward(activation1.output) #layer 2 weights and biases
    activation2.forward(dense2.output) #activation softmax 

    #take the output from the softmax and return the loss (measure of inaccuracy)
    loss = loss_function.calculate(activation2.output, y)

    #calculate the accuracy from the output of activation 2 and the targets
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    #if loss is smaller then print and save weights and biases
    if loss < lowest_loss:
        print('New set of weight found - Iteration: ', i, 
            '\nLoss: ', loss, '\nAccuracy: ', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()