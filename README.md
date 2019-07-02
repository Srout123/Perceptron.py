In this model, we have n binary inputs (usually given as a vector) and exactly the same number of weights W_1, ..., W_n.
We multiply these together and sum them up. We denote this as z and call it the pre-activation.
We can re-write this as an inner product for succinctness.) There is another term, called the bias, that is just a constant factor.
After taking the weighted sum, we apply an activation function, \sigma, to this and produce an activation a. 
The activation function for perceptrons is sometimes called a step function because, if we were to plot it, it would look like a stair.


There are two types of Perceptrons: Single layer and Multilayer.

Single layer Perceptrons can learn only linearly separable patterns.

Multilayer Perceptrons or feedforward neural networks with two or more layers have the greater processing power.

The Perceptron algorithm learns the weights for the input signals in order to draw a linear decision boundary.

This enables you to distinguish between the two linearly separable classes +1 and -1.

Note: Supervised Learning is a type of Machine Learning used to learn models from labeled training data.
It enables output prediction for future or unseen data.

Let us focus on the Perceptron Learning Rule in the next section.

Perceptron Learning Rule
Perceptron Learning Rule states that the algorithm would automatically learn the optimal weight coefficients.
The input features are then multiplied with these weights to determine if a neuron fires or not.
The Perceptron receives multiple input signals, and if the sum of the input signals exceeds a certain threshold,
it either outputs a signal or does not return an output. In the context of supervised learning and classification,
this can then be used to predict the class of a sample.

In the next section, let us focus on the perceptron function.

Perceptron Function
Perceptron is a function that maps its input “x,”
which is multiplied with the learned weight coefficient;
an output value ”f(x)”is generated.
In the equation given above:

“w” = vector of real-valued weights

“b” = bias (an element that adjusts the boundary away from origin without any dependence on the input value)

Activation Functions of Perceptron

The activation function applies a step rule (convert the numerical output into +1 or -1) to check if the output of the weighting function is greater than zero or not.

“x” = vector of input x values

“m” = number of inputs to the Perceptron

The output can be represented as “1” or “0.”  It can also be represented as “1” or “-1” depending on which activation function is used.

Let us learn the inputs of a perceptron in the next section.

Inputs of a Perceptron
A Perceptron accepts inputs, moderates them with certain weight values, 
then applies the transformation function to output the final result. The above below shows a Perceptron with a Boolean output.

A Boolean output is based on inputs such as salaried, married, age, past credit profile, etc.
It has only two values: Yes and No or True and False.
The summation function “∑” multiplies all inputs of “x” by weights “w” and then adds them up

Output of Perceptron
Perceptron with a Boolean output:

Inputs: x1…xn

Output: o(x1….xn)
Weights: wi=> contribution of input xi to the Perceptron output;

w0=> bias or threshold

If ∑w.x > 0, output is +1, else -1. The neuron gets triggered only when weighted input reaches a certain threshold value.

An output of +1 specifies that the neuron is triggered. An output of -1 specifies that the neuron did not get triggered.

“sgn” stands for sign function with output +1 or -1.

Error in Perceptron
In the Perceptron Learning Rule, the predicted output is compared with the known output. If it does not match, the error is propagated backward to allow weight adjustment to happen.

Let us discuss the decision function of Perceptron in the next section.

Perceptron: Decision Function
A decision function φ(z) of Perceptron is defined to take a linear combination of x and w vectors.

