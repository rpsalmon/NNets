'''the output is not a category it is a confidence score
based on the training data - not a regression
mean absolute error allows you to measure accuracy
categorical cross entropy = -sum(log(predicted value * target label index))
one-hot encoding, number of classed determines length of vector and the
one-hot value indicates the position the target lands in the vector'''

'''what is a logrithm - solving for x
e**x = b'''
import numpy as np
b = 5.2
#print (np.log(b))
import math
#print (math.e**(np.log(b))) #equivalent to b

softmax_out = [0.7, 0.1, 0.2]
target_class = 0
target_out = [1, 0, 0]

loss = -(math.log(softmax_out[0])*target_out[0]+\
    math.log(softmax_out[1])*target_out[1]+\
        math.log(softmax_out[2])*target_out[2])

print(loss)

'''categorical cross entropy returns a value inverse to the confidence
of the target value - research additiional loss functions'''