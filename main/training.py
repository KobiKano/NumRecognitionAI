#  This file is used to train Neural Network using MNIST dataset
import numpy as np
from tqdm import tqdm


#  Function to start training neural network
def train(network, images, labels):
    # assume network initialized
    # ask user to define parameters of training
    learning_rate = network.learning_rate
    num_back_prop = 1
    epoch_size = 1000
    weight_range = network.weight_range
    bias_range = network.bias_range
    while True:
        user_in = input("Define Training Parameters\n"
                        "1) learning rate - Current: {}\n"
                        "2) number back prop - Current: {}\n"
                        "3) epoch size - Current: {}\n"
                        "4) weight range - Current: {}\n"
                        "5) bias range - Current: {}\n"
                        "6) continue\n".format(learning_rate, num_back_prop, epoch_size,
                                               weight_range, bias_range))

        match int(user_in):
            case 1:
                learning_rate = float(input("Enter Float value\n"))
            case 2:
                num_back_prop = int(input("Enter Int value\n"))
            case 3:
                epoch_size = int(input("Enter Int value\n"))
            case 4:
                weight_range = float(input("Enter Float value\n"))
            case 5:
                bias_range = float(input("Enter Float value\n"))
            case default:
                break

        # set learning rate and weight/bias range
        network.set_learning_rate(learning_rate)
        network.reset_weights_biases(weight_range, bias_range)

    num_success = 0
    epoch_num = 0
    # iterate through all images
    for i in tqdm(range(len(images)), desc="Training..."):
        # print("training round {}\nLabel: {}\n".format(i, labels[i]))

        # check if next epoch and print accuracy
        if i % epoch_size == 0 and i != 0:
            epoch_num += 1
            print("Accuracy of Epoch {} : {}\n".format(epoch_num, float(num_success) / float(epoch_size)))
            num_success = 0
            # start new progress bar

        # form input list
        inputs = []
        for row in images[i]:
            inputs.extend(row)
        # normalize inputs between 0.0 and 1.0
        inputs = np.array(inputs)
        inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))
        inputs = inputs.tolist()

        # form desired output list
        desired = np.zeros(10)
        desired[labels[i]] = 1

        # send training data to network
        if network.train(inputs, desired, num_back_prop):
            num_success += 1

    # indicate training finished and save network
    user_in = input("Training Complete!\n"
                    "Save Network (y/n)\n")
    if user_in == 'y':
        network.save_weights_biases("NumRecognitionAI/Data/Network/network.csv")

    # return from function
    return
