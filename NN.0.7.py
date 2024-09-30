'''incorporating the loss function (categorical cross entropy) 
into the NN'''

softmax_output = [[0.7, 0.1, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]]

'''sparse target vector for the time being in place of one-hot
classes:
0 - dog
1 - cat
2 - human
because we are using a softmax the positions of the max values
are [0, 1, 1]'''

class_target = [0, 1, 1] 
'''we can also use a 2-d array'''
class_target = [[1, 0 ,0],
                [0, 1, 0],
                [0, 1, 0]]

'''zip the lists together and iterate softax at the index of the target'''
#for targ_idx, dist in zip(class_target, softmax_output):
#    print(dist[targ_idx])

import numpy as np
from numpy.core.fromnumeric import shape

softmax_output = np.array(softmax_output) #turn it into an array and index
class_target = np.array(class_target)
#print(softmax_output[[0,1,2], class_target])
'''replace the hardcoded range with python code'''
softmax_target = softmax_output[range(len(softmax_output)), class_target]
'''categorical class entropy for the loss'''
losses = -np.log(softmax_target)
batch_losses = np.mean(losses)
samples = len(softmax_output)
print(range(samples))
#print(batch_losses)
'''because -log(0) is infinite we will arrive at an error if the 
target value has a confidence interval of zero and you will need
to clip the values (limit the values)
y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)'''
y_pred_clip = np.clip(softmax_output, 1e-7, 1-1e-7)
#print(-np.log(1-1e-7))
#print(y_pred_clip)
if len(class_target.shape) == 1:
    confidences = y_pred_clip[range(samples), class_target]
    print(confidences)
elif len(class_target.shape) == 2:
    confidences = np.sum(y_pred_clip*class_target, axis =1)
    print(confidences)
neg_log_likely = -np.log(confidences)
print(neg_log_likely)
'''so what we have above is clipping the results between a given range
and if statement verifying that what is being passed is scaler
then for the range of samples return each value in the targeted positions
if the targets are a 2-dim array of one-hot encoded vectors and confidences
are equivalent to softmanx_targets earlier in the code'''

'''this will be the basis for measuring accuracy for a known result to 
validate the accuracy of a model before back propogation'''
prediction = np.argmax(softmax_output, axis = 1)
accuracy = np.mean(prediction == class_target)
