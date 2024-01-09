# NumRecognitionAI
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<img alt="GitHub contributors" src="https://img.shields.io/github/contributors/KobiKano/NumRecognitionAI?color=green">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/KobiKano/NumRecognitionAI?color=blue">

# Project Description
    The goal of this project was for me to really test the neural network implementation I designed in my MusicAI project.
    It allowed me to debug the problems I had in the musicAI network a lot easier due to working with an easier dataset,
    that dataset being the MNIST Dataset

    You can find the dataset at https://www.kaggle.com/datasets/hojjatk/mnist-dataset

# Features
    - Fully customizable network parameters before training
    - Added my own network to test against
    - Epoch and accuracy checkers during training cycle

# Startup
    - Start the main application and input desired network size (input and output are not changeable)
    - Leave network size as default to test my network
# Training
    - Define training parameters for learning rate, number of back propagations, epoch size, weight/bias ranges
    - Learning Rate dictates the "speed" at which the network adjusts due to the gradient of the error with respect to a weight
    - The number of back propagations dictates the number of times the program will propagate back the error per training entry
    - Epoch size dictates the number of inputs used in any given accuracy measurement
    - Weight/bias range dictates the range at which the biases or weights can lie when first initialized to random values.  The range is from [-weight/bias range to weight/bias range]
# Testing
    - Load the existing saved network from csv to test my network under default parameters
    - Or test your own network (as long as it has been trained)
    - It works on an untrained network as well, but there really isn't much point in doing that

# Network Implementation

### Overview
    This is a relatively simple network implementation, the main points are:
    - The activation function over all nodes is a sigmoid function
    - The network uses a static learning rate
    - All weights and biases are initialized in the same way
    - The backpropgation algorithm used is stochasitc descent over the cost function with respect to the weights and biases
    - The cost function used is the means squared error

### Gradient Descent
    The derrivations for the gradients are quite simple, documentation for the exact formulas used can be found online, but the important parts to understand are:
    - The gradients are found using the chain rule with the gradient with despect to a weight or bias being split into multiple partial derrivates
    - Another important thing to consider is that any change to a hidden layer affects all connected nodes, however as long as the local gradients of the higher(closer to output) layer are known we only need to consider the higher layer

# Takeaways
    The network learns quite well, something to note is that I tested a larger network with 3 hidden layers and 100 nodes per layer
    This network performed worse given all 60000 training inputs than the current default
    The current default network in the csv has an accuracy of about 90%