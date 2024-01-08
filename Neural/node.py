import numpy as np


def sigmoid(x):
    output = 0
    # avoid overflow by setting output to 1 since the sigmoid of a large number is essentially one
    try:
        output = 1 / (1 + np.exp(-x))
    except OverflowError:
        output = 0

    return output


def sigmoid_prime(x):
    # assume x is sigmoid result
    return x * (1 - x)


class Node:
    # initialize fields
    weights = []
    bias = 0.0
    last_input = []
    last_sum = 0.0
    sigmoid_prime = 0.0

    def __init__(self, weights: list, bias: float):
        self.weights = weights
        self.bias = bias

    def get_sum(self, inputs : list):
        # make sure inputs same size as weights
        if len(inputs) != len(self.weights):
            raise Exception(f"Inputs to node not the same size as node weights!\n"
                            f"Inputs: {len(inputs)}\n "
                            f"Weights: {len(self.weights)}")
        # sum all values
        s = np.dot(inputs, self.weights) + self.bias

        self.last_input = inputs.copy()

        # return sigmoid squishification of sum
        self.last_sum = sigmoid(s)
        self.sigmoid_prime = sigmoid_prime(self.last_sum)
        return self.last_sum

    def set_weights(self, weights: list):
        self.weights = weights

    def set_bias(self, bias: float):
        self.bias = bias
