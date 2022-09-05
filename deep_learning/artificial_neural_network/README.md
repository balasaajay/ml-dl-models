# Artificial Neural Networks:
## The neuron
- neuron gets some input signal (can be from another neuron) and has output signal
- also reffered to as node
- can be part of input layer, output or hidden layer
- Input layers input values should be normalized/standardized
- Output can be continuous, binary, categorical 
- connections between neuraons has weights. Determines the significance of a connection
- What happens in neuron:
  1. adds up all (weight_i*input_i)
  2. applies activation function

## Activation functions
- executed in Neuron
- types:
  1. Threshold function: 
      if val >= 0 => passes 1
      if val < 0  => passes 0
  2. sigmoid function (can be used in output layer)
      fn = 1/(1+exp(-x))
      gradual progression unlike threshold function
  3. Rectified Linear Unit (ReLU)
      fn = max(x, 0)
      most used
  4. Hyperbolic Tangent (tanh)
      fn = (1-e(-2x))/(1+e(-2x))
      goes below zero
  5. softmax (usually used in Output layer)
- For binary input variable - threshold or sigmoid can be used

## How Neural networks work?
- creates the hidden layer inputs as the combinations of the input layer neuron outputs

## How NNets learn?
- perceptron
- calculates cost function for every training row value and feeds it back to the NN and updates weights
- Epoch
- Adjust weights after every epoch (Batch Gradient Descent, mini-batch gradient descent)
- Goal is to minimize cost function
- This whole process is called back propagation

## Gradient descent
- ways to get to minimum cost function 
1. Brute force - try randomly for few weights and find the minimum cost function. Disadvatage is curse of dimensionality -> solve this with gradient descent
2. gradient descent

## Stochastic Gradient descent
- If cost function has multiple minima, use stochastic gradient descent
- adjust weights after every single row of training data to minimize cost function
- it is random

## Backpropagation
- Based on the cost function adjust all weights to minimize cost function

## Training ANN with SGD:
1. randomly initialize weights to small numbers close to 0 (but not 0)
2. input the first observation in input layer, each feature is a node in input layer
3. Forward propagation:(left to right) propagate activation until getting predicted y
4. Compare predicted result to actual result and measure the generated error
5. Back propagation: (right to left) error is back propagated and weights are updated.Learning rate decides how much weights are updated.
6. Repeat steps 1-5 after each observation (Reinforcement learning) or only after a batch of obs (Natch learning)
7. When whole training set passed throgh the NN => one epoch. Redo more epochs

