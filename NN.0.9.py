'''Implementation of backwards propogation 
in 0.8 we learned that we can randomly guess but there 
are limitations in efficiency'''

import numpy as np
np.random.seed(0)
#the spiral data set we are working to predict
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

print("data created")

'''the dense layer, hidden layer, node, etc whatever you want to call it
we were calling this layer_dense but will be calling it a node from here 
on forward'''
class Node:
#initialize the weights and biases
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_nuerons) 
        self.biases = np.zeros((1, n_nuerons)) 
#forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
#backward pass
    def backward(self, dvalues):
    #gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    #gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activataion_relu:
#forward pass
    def forward(self, inputs):
        #store the inputs
        self.inputs = inputs
        #calculate the outputs from the inputs
        self.output = np.maximum(0, inputs)
    
#backward pass
    def backward(self, dvalues):
    #because it will be modified we want to store a copy of the original
        self.dinputs = dvalues.copy()
    #zero gradient where inputs are negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_softmax:
# forward pass
    def forward(self, inputs):
    # store the input values
        self.inputs = inputs
    # get probabilities or expected values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    #normalize each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
    #create an empty unitinitialized array 
        self.dinputs = np.empty_like(dvalues)
    #enumerate the outputs and gradients
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output.dvalues)):
        #flatten output array
            single_output = single_output.reshape(-1,1)
        #calculate the Jacobian matrix of the output
            j_matrix = np.diagflat(single_output) \
                - np.dot(single_output, single_output.T)
            #calculate the sample wise gradient and add to array of sample gradients
            self.dinputs[index] = np.dot(j_matrix, single_dvalues)



'''gradient refers to the amount of influence the individual component
has on the output
it is an optimization function for finding the local minimum (error)
of the different functions applied throughout the model
the descent is the act of reducing the amount of error
b = a - w(derivative of f(a)) 
where a is beginning value minus weighted derivative of the 
activation function'''

# the loss function classes
class Loss:
# calculate the regularized losses, how wrong is it
    def calculate(self, output, y):
    #calculate the sample loss
        sample_loss = self.forward(output, y)
    #calculate the mean loss
        mean_loss = np.mean(sample_loss)

        return mean_loss

class Cat_cross_ent(Loss):
    def forward(self, y_pred, y_true):
    # how many samples in each batch
        samples = len(y_pred)
    #clip the data on both sides to prevent a divisble by 0 error
        y_pred_c = np.clip(y_pred, 1e-7, 1-1e-7)
        
    # target value probabilities
    # categorical scaler values
        if len(y_true.shape) == 1:
            confidences = y_pred_c[range(samples), y_true]

    # for one-hot encoded labels
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_c * y_true, axis=1)

    # losses 
        neg_log_likely = -np.log(confidences)
        return neg_log_likely

    def backward(self, dvalues, y_true):
    # how many samples
        samples = len(dvalues)
    # how many labels in each sample
        labels = len(dvalues[0]) #using the first sample

    #takes vector labels and turns into one-hot encoded vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
    #caluclate the gradients
        self.dinputs = -y_true / dvalues
    # normalize the gradients
        self.dinputs = self.dinputs / samples

class Forward():
    def __init__(self):
        self.activation = Activation_softmax()
        self.loss = Cat_cross_ent()
    def forward(self, inputs, y_true):
    # set output to layer activation function
        self.activation.forward(inputs)
        self.output = self.activation.output
    # calculate and return the loss value
        return self.loss.calculate(self.output, y_true)

class Backward():
    def backward(self, dvalues, y_true):
    # how many samples 
        samples = len(dvalues)

    #if labels are one-hot encoded turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axies=1)

    # copy and modify the values
        self.dinputs = dvalues.copy()
    # calculate the gradient
        self.dinputs[range(samples), y_true] -= 1
    # normalize the gradient 
        self.dinputs = self.dinputs / samples
        
'''#--------------------------------------------------
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_softmax()
        self.loss = Cat_cross_ent()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)


    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

#--------------------------------------------------'''

X , y = create_data(100, 3)

node1 = Node(2,3) #2 inputs and 3 outputs
node2 = Node(3,3) #3 inputs from previous layer and 3 ouptuts 

activation1 = Activataion_relu() #our activation function(s)
activation2 = Activation_softmax() #output function

loss_function = Cat_cross_ent()

forward_ = Forward()
back = Backward()
#forward pass of the data through the node
node1.forward(X)
#pass the outut from the first node through the activation
activation1.forward(node1.output)
#pass the result of the function through the second node
node2.forward(activation1.output)
#returns the softmax results from the results of the second node
activation2.forward(node2.output)
#calculates the loss from the output of the softmax function
loss = forward_.forward(activation2.output, y)
print("Activation sample: ", activation2.output[:5])
print("Loss: ", loss)

# calculate the accuracy from the output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(forward_.forward, axis = 1)
if len(y.shape) == 2:
    y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions==y)
print('accuracy: ', accuracy)

#the backwards pass
'''this is laid out in the opposite order of the forward pass
as it will process the data in *reverse* order'''
back.backward(activation2.output, y)
node2.backward(back.dinputs)
activation1.backward(node2.dinputs)
node1.backward(activation1.dinputs)


#print the gradients
print("Node1W: ", node1.dweights)
print("Node1B: ", node1.dbiases)
print("Node1W: ", node2.dweights)
print("Node1B: ", node2.dbiases)

'''#--------------------------------------
# Create Dense layer with 2 input features and 3 output values
dense1 = Node(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activataion_relu()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Node(3, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)
# Let's see output of the first few samples:
print(loss_activation.output[:5])

# Print loss value
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

# Print accuracy
print('acc:', accuracy)

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)'''