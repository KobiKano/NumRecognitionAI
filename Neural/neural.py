import pandas as pd
import NumRecognitionAI.Neural.node as node
import numpy as np
import numpy.random as rand
import math
import ast
import csv


# This file contains all functions for the initialization of a neural network

# random filler to allow negative values
def fill(num):
    output = []
    for i in range(num):
        output.append(rand.uniform(-1.0, 1.0))
    return output


# this is the network class that will house all functions for the neural network
class Network:

    def __init__(self, num_layers, layer_size, num_inputs, num_outputs):
        # initialize fields
        self.network = []
        self.layers = 0
        self.learning_rate = 1

        # initialize layers
        self.layers = num_layers
        for i in range(num_layers):
            self.network.append([])
            # check if input layer
            if i == 0:
                for j in range(layer_size):
                    self.network[i].append(node.Node(fill(num_inputs), rand.uniform(-1.0, 1.0)))
                continue
            for j in range(layer_size):
                self.network[i].append(node.Node(fill(layer_size), rand.uniform(-1.0, 1.0)))

        # initialize output layer to all zeros
        self.network.append([])
        for j in range(num_outputs):
            self.network[num_layers].append(node.Node(np.zeros(layer_size, dtype=float).tolist(), 0.0))


        print(f"Length of network: {len(self.network)}")
        print(f"Length of output layer: {len(self.network[self.layers])}")

    def set_learning_rate(self, rate):
        self.learning_rate = rate

    def set_weights_biases(self, weights, biases):
        # assume weights are 3-dimensional array
        index = 0
        for i in range(len(self.network)):  # layer
            for j in range(len(self.network[i])):  # node
                self.network[i][j].set_weights(ast.literal_eval(weights[index]))
                self.network[i][j].set_bias(biases[index])
                index += 1

    def save_weights_biases(self, path):
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

    def train(self, inputs, desired, num_back):
        # train on array of inputs
        outputs = self.forward_prop(inputs)
        # check outputs and compare to desired cost
        cost = 0
        for i in range(len(outputs)):
            # print(f"Desired: {desired[i]}\nOutput: {outputs[i]}")
            cost += math.pow((outputs[i] - desired[i]), 2)

        print("Cost of last train: " + str(cost))
        # send cost feedback backwards to adjust weights and biases
        for i in range(num_back):
            self.backward_prop(desired, inputs)  # back prop a user defined num times

    def forward_prop(self, inputs):
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

    def backward_prop(self, desired, inputs):
        # find gradients at output layer
        prev_gradients = []
        for i in range(len(self.network[self.layers])):
            local_gradient = 2*(self.network[self.layers][i].last_sum - desired[i])
            prev_gradients.append(local_gradient)

            # find new weights and biases
            for j in range(len(self.network[self.layers][i].weights)):
                self.network[self.layers][i].weights[j] -= local_gradient * self.learning_rate * self.network[self.layers - 1][j].last_sum

            self.network[self.layers][i].bias -= local_gradient * self.learning_rate

        # find gradients of lower layers
        i = self.layers - 1
        while i != -1:  # iterate through layers backwards
            gradients = []
            for j in range(len(self.network[i])):  # iterate through each node in this layer
                # find local gradient by summing over values in L + 1 layer
                local_gradient = 0
                # print(len(self.network[i + 1]))
                for k in range(len(self.network[i + 1])):  # iterate through all nodes in next layer
                    # add product of weight from this node to every node in L + 1
                    #                sigmoid prime of other node
                    #                calculated local gradient at other node
                    local_gradient += self.network[i + 1][k].weights[j] * self.network[i + 1][k].sigmoid_prime * prev_gradients[k]

                gradients.append(local_gradient)

                # find new weights and biases
                # check if on lowest layer, if so use input values to adjust weights, else use prev layer output
                # iterate through each weight and adjust
                if i == 0:
                    for k in range(len(self.network[i][j].weights)):
                        self.network[i][j].weights[k] -= local_gradient * self.learning_rate * inputs[k]

                else:
                    for k in range(len(self.network[i][j].weights)):
                        self.network[i][j].weights[k] -= local_gradient * self.learning_rate * self.network[i - 1][k].last_sum

                self.network[i][j].bias -= local_gradient * self.learning_rate

            prev_gradients = gradients.copy()
            i -= 1

        return
