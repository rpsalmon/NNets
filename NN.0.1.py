'''initial basics of how a NN is built
the basics of inputs weights and biases 
one nueron with 3+ inputs'''
inputs = [1, 2, 3, 2.5] #values from sensors or some other source
#weights = [0.2, 0.8, -0.5, 1.0] #a weight set for a unique nueron
#bias = 2
'''output = inputs[0]*weights[0]\
            +inputs[1]*weights[1]\
            +inputs[2]*weights[2]\
            +bias'''
#print(output)
'''to create additional nuerons we need 
the equivalent numebr weight sets and biases'''
#weights2 = [0.5, -0.91, 0.26, -0.5] #a weight set for a unique nueron
#bias2 = 3
#weights3 = [-0.26, -0.27, 0.17, 0.87] #a weight set for a unique nueron
#bias3 = 0.5 
'''output = [inputs[0]*weights[0]\
            +inputs[1]*weights[1]\
            +inputs[2]*weights[2]\
            +inputs[3]*weights[3]\
            +bias,  #nueron one
            inputs[0]*weights2[0]\
            +inputs[1]*weights2[1]\
            +inputs[2]*weights2[2]\
            +inputs[3]*weights2[3]\
            +bias2, #nueron two
            inputs[0]*weights3[0]\
            +inputs[1]*weights3[1]\
            +inputs[2]*weights3[2]\
            +inputs[3]*weights3[3]\
            +bias3] #nueron three
print(output)'''

'''to simplify the above we are going to make
a list of lists for weights and a list of biases'''
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = [] #output of current layer
'''what we have below is a for loop in a for loop
that performs the previous operation, the zip function
ties the values from the 2 separate lists into a tuple in 
this case the first list in weights is paired with the first 
value in biases, second list with second value etc etc'''
for nueron_weights, nueron_bias in zip(weights, biases):
    nueron_output = 0 #output of given nueron
    for n_input, weight in zip(inputs, nueron_weights):
        nueron_output += n_input*weight 
        '''var1+=var2 is same as var1=var1+var2'''
    nueron_output += nueron_bias
    layer_outputs.append(nueron_output)

print(layer_outputs)