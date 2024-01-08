import pandas as pd
import NumRecognitionAI.Neural.node as node
import numpy.random as rand
import math
import ast
import csv


# This file contains all functions for the initialization of a neural network

# random filler to allow negative values
def fill(num: int, weight_range: float):
    output = []
    for i in range(num):
        output.append(rand.uniform(-weight_range, weight_range))
    return output


# this is the network class that will house all functions for the neural network
# this network assumes all data inputs are normalized between 0.0 and 1.0
class Network:

    def __init__(self, num_layers: int, layer_size: int, num_inputs: int, num_outputs: int,
                 weight_range: float, bias_range: float):
        # initialize fields
        self.network = []
        self.layers = 0
        self.learning_rate = 1.0
        # save for external modification
        self.layer_size = layer_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight_range = weight_range
        self.bias_range = bias_range

        # initialize layers
        self.layers = num_layers
        self.reset_weights_biases(weight_range, bias_range)

    def set_learning_rate(self, rate: float):
        self.learning_rate = rate

    def reset_weights_biases(self, weight_range: float, bias_range: float):
        self.network = []
        # initialize layers
        for i in range(self.layers):
            self.network.append([])
            # check if input layer
            if i == 0:
                for j in range(self.layer_size):
                    self.network[i].append(node.Node(fill(self.num_inputs, weight_range),
                                                     rand.uniform(-bias_range, bias_range)))
                continue
            for j in range(self.layer_size):
                self.network[i].append(
                    node.Node(fill(self.layer_size, weight_range), rand.uniform(-bias_range, bias_range)))

        # initialize output layer
        self.network.append([])
        for j in range(self.num_outputs):
            self.network[self.layers].append(node.Node(fill(self.layer_size, weight_range),
                                                       rand.uniform(-bias_range, bias_range)))

    def set_weights_biases(self, weights: list, biases: list):
        # assume weights are 3-dimensional array
        index = 0
        for i in range(len(self.network)):  # layer
            for j in range(len(self.network[i])):  # node
                self.network[i][j].set_weights(ast.literal_eval(weights[index]))
                self.network[i][j].set_bias(biases[index])
                index += 1

    def save_weights_biases(self, path: str):
        data = {
            "layer": [],
            "node": [],
            "weights": [],
            "bias": []
        }
        # save weights and biases in csv
        for i in range(len(self.network)):  # layer
            for j in range(len(self.network[i])):  # node
                data["layer"].append(i)
                data["node"].append(j)
                data["weights"].append(self.network[i][j].weights)
                data["bias"].append(self.network[i][j].bias)
        df = pd.DataFrame(data=data)
        df.to_csv(path, quoting=csv.QUOTE_ALL)

    def print_cost(self, outputs, desired):
        # check outputs and compare to desired cost
        cost = 0
        for i in range(len(outputs)):
            # print(f"Desired: {desired[i]}\nOutput: {outputs[i]}")
            cost += math.pow((outputs[i] - desired[i]), 2)
        cost = cost / len(outputs)  # mean squared error
        print("Cost of last train: " + str(cost))

    # returns boolean success value if network correctly identifies input
    def train(self, inputs: list, desired: list, num_back_prop: int):
        # train on array of inputs
        outputs = self.forward_prop(inputs)

        # print cost if debugging
        # self.print_cost(outputs, desired)

        # check if network identified correct value from input
        index = outputs.index(max(outputs))
        success = desired[index] == 1.0

        # send cost feedback backwards to adjust weights and biases
        for i in range(num_back_prop):
            self.backward_prop(desired, inputs)  # back prop a user defined num times

        # if correct identified return true, otherwise return false
        return success

    def forward_prop(self, inputs: list):
        outputs = []
        # go through each node in first layer and feed inputs and save outputs
        for j in range(len(self.network[0])):
            outputs.append(self.network[0][j].get_sum(inputs))

        # go through each node in each following layer and find outputs
        for i in range(1, self.layers + 1):
            next_outputs = []
            # go through each node and feed inputs and save outputs
            for j in range(len(self.network[i])):
                next_outputs.append(self.network[i][j].get_sum(outputs))
            outputs = next_outputs.copy()

        return outputs

    def backward_prop(self, desired: list, inputs: list):
        # find gradients at output layer
        prev_gradients = []
        for layer_node in self.network[self.layers]:
            index = self.network[self.layers].index(layer_node)  # save index value for future use

            # this is the partial derivative of the cost function with respect to this specific output node
            local_gradient = (layer_node.last_sum - desired[index]) * (2 / len(self.network[self.layers]))
            prev_gradients.append(local_gradient)  # save local gradients for use in calculating future cost gradients

            # find new weights and biases
            delta = local_gradient * layer_node.sigmoid_prime
            for j in range(len(layer_node.weights)):
                layer_node.weights[j] -= self.learning_rate * delta * self.network[self.layers - 1][j].last_sum

            layer_node.bias -= self.learning_rate * delta

        # hidden layer nodes are more complicated with the need to account for their effect on following nodes
        # find gradients of lower layers
        i = self.layers - 1
        while i != -1:  # iterate through layers backwards
            gradients = []
            for j in range(len(self.network[i])):  # iterate through each node in this layer
                # find local gradient by finding partial derivative of cost with respect to activation on current node
                #                   This is done by accounting for all possible effects a change in this activation
                #                   can have on all connected nodes in further layers
                local_gradient = 0
                for k in range(len(self.network[i + 1])):  # iterate through all nodes in next layer
                    # add product of weight from this node to every node in L + 1
                    #                sigmoid prime of other node
                    #                calculated local gradient at other node
                    local_gradient += (self.network[i + 1][k].weights[j] * self.network[i + 1][k].sigmoid_prime
                                       * prev_gradients[k])

                gradients.append(local_gradient)

                # find new weights and biases
                # check if on lowest layer, if so use input values to adjust weights, else use prev layer output
                # iterate through each weight and adjust
                delta = local_gradient * self.network[i][j].sigmoid_prime  # change dependent on specific node
                if i == 0:
                    for k in range(len(self.network[i][j].weights)):
                        self.network[i][j].weights[k] -= self.learning_rate * delta * inputs[k]

                else:
                    for k in range(len(self.network[i][j].weights)):
                        self.network[i][j].weights[k] -= self.learning_rate * delta * self.network[i - 1][k].last_sum

                # bias is only dependent on node, therefore layer level doesn't matter
                self.network[i][j].bias -= self.learning_rate * delta

            prev_gradients = gradients.copy()
            i -= 1

        return
