# Perceptron.py
In this model, we have n binary inputs (usually given as a vector) and exactly the same number of weights W_1, ..., W_n.
We multiply these together and sum them up. We denote this as z and call it the pre-activation.
We can re-write this as an inner product for succinctness.) There is another term, called the bias, that is just a constant factor.
After taking the weighted sum, we apply an activation function, \sigma, to this and produce an activation a. 
The activation function for perceptrons is sometimes called a step function because, if we were to plot it, it would look like a stair.


import numpy as np

#Input array
print ("Enter a square matrix")
n = int(input(("Enter number of columns in the matrix:")).strip())
m = int(input(("Enter number of rows in the matrix:")).strip())
print("Enter the matrix by space separated")
X = np.array([[0]*n for _ in range(m)])
for i in range(n):
    X[i] = [int(j) for j in input(("Enter Row Value and columns Value:")).strip().split(" ")]
print ("The Entered Inputs Nuerons are:")
print(X)

y=np.array([[0],[1]])
hidden = int(input('How many hidden layer nuerons: '))
output = int(input('How many output layer nuerons: '))
    
#Sigmoid Function


def sigmoid (x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function


def derivatives_sigmoid(x):
	return x * (1 - x)

#Variable initialization


epoch=5000 #Setting training iterations
lr=0.1
inputlayer_neurons = X.shape[0] #number of features in data set
hiddenlayer_neurons = hidden #number of hidden layers neurons
output_neurons = output #number of neurons at output layer

#weight and bias initialization
W1=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
#Random Weight Generated
B1=np.random.uniform(size=(1,hiddenlayer_neurons))
#random baised weight generated
W2=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
#Random Output Weight Generated
B2 =np.random.uniform(size=(1,output_neurons))
#Random Baised Output Weight Generated

print("The randomly Generated Weights from inputlayer_neurons to hiddenlayer_neurons are:")
print(W1)

print("The randomly Generated Baised Weights From inputlayer_neurons to hiddenlayer_neurons are:")
print(B1)

print("The randomly Generated Weights From the hiddenlayer_activations to output_neurons are:")
print(W2)

print("The randomly Generated Baised Weights From the hiddenlayer_activations to output_neurons are:")
print(B2)


for i in range(epoch):
	hidden_layer_input1=np.dot(X,W1)
hidden_layer_input=hidden_layer_input1 + B1
hiddenlayer_activations = sigmoid(hidden_layer_input)
print("The hiddenlayer_activations:-",hiddenlayer_activations)
output_layer_input1=np.dot(hiddenlayer_activations,W2)
output_layer_input= output_layer_input1+ B2
output = sigmoid(output_layer_input)

E = y-output
slope_output_layer = derivatives_sigmoid(output)
slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
d_output = E * slope_output_layer
Error_at_hidden_layer = d_output.dot(W2.T)
d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
W2+= hiddenlayer_activations.T.dot(d_output) *lr
B2+= np.sum(d_output, axis=0,keepdims=True) *lr
W1+= X.T.dot(d_hiddenlayer) *lr
B1+= np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr


print("The Final Output:",output)




