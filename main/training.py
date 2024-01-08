#  This file is used to train Neural Network using MNIST dataset
import numpy as np


#  Function to start training neural network
def train(network, images, labels):
    # assume network initialized

    # iterate through all images
    for i in range(len(images)):
        print("training round {}\n"
              "Label: {}\n".format(i, labels[i]))

        # form input list
        inputs = []
        for row in images[i]:
            inputs.extend(row)
        # normalize inputs between 0.0 and 1.0
        inputs = np.array(inputs)
        inputs = (inputs - np.min(inputs))/(np.max(inputs) - np.min(inputs))
        inputs = inputs.tolist()

        # form desired output list
        desired = np.zeros(10)
        desired[labels[i]] = 1

        # send training data to network
        network.train(inputs, desired, 1)

    # indicate training finished and save network
    user_in = input("Training Complete!\n"
                    "Save Network (y/n)\n")
    if user_in == 'y':
        network.save_weights_biases("NumRecognitionAI/Data/Network/network.csv")

    # return from function
    return
