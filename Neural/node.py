import math


def sigmoid(x):
    output = 0
    # avoid overflow by setting output to 1 since the sigmoid of a large number is essentially one
    try:
        output = 1/(1+math.exp(-x))
    except OverflowError:
        output = 1

    return output


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Node:
    # initialize fields
    weights = []
    bias = 0
    last_input = []
    last_sum = 0
    sigmoid_prime = 0
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def get_sum(self, inputs):
        # make sure inputs same size as weights
        if len(inputs) != len(self.weights):
            raise Exception(f"Inputs to node not the same size as node weights!\n"
                            f"Inputs: {len(inputs)}\n "
                            f"Weights: {len(self.weights)}")
        # sum all values
        s = 0
        for i in range(len(self.weights)):
            s += self.weights[i] * inputs[i]
        # add bias
        s += self.bias

        self.last_input = inputs.copy()

        # return sigmoid squishification of sum
        self.last_sum = sigmoid(s)
        self.sigmoid_prime = sigmoid_prime(s)
        return self.last_sum

    def set_weights(self, weights):
        self.weights = weights
    def set_bias(self, bias):
        self.bias = bias

