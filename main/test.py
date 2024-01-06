#  This file is used to test trained network using MNIST dataset
import pandas as pd
import random as rand
import matplotlib.pyplot as plt

#  This function tests the network
def test(network, images, labels):
    # assume network initialized

    # ask user if they want to load from trained csv or use current network
    user_in = input("Input Expected\n"
                    "1) Load from csv\n"
                    "2) Use Current Network\n")

    if int(user_in) == 1:
        df = pd.read_csv("NumRecognitionAI/Data/Network/network.csv")
        network.set_weights_biases(df["weights"].tolist(), df["bias"])

    while True:
        # pick random image from testing set
        r = rand.randint(0, len(labels) - 1)

        # show image at random index
        plt.imshow(images[r], cmap=plt.cm.gray)

        # propagate through neural network
        inputs = []
        for row in images[r]:
            inputs.extend(row)
        output = network.forward_prop(inputs)

        # check output
        print("Desired Output: {}\n"
              "Resulting Output:{}\n".format(labels[r], output.index(max(output))))

        # ask if user wants to continue
        user_in = input("Continue (y/n)\n")
        if user_in != 'y':
            return
