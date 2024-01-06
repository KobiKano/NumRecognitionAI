#  Main file for project
import matplotlib.pyplot as plt
from os.path import join
from NumRecognitionAI.MNIST.MNIST_READER import MnistDataloader
from NumRecognitionAI.Neural.neural import Network
from training import train
from test import test
#  Will allow for testing of neural network generated in training file
if __name__ == '__main__':
    #
    # Set file paths based on added MNIST Datasets
    #
    training_images_filepath = 'NumRecognitionAI/Data/Dataset/train-images-idx3-ubyte/train-images-idx3-ubyte'
    training_labels_filepath = 'NumRecognitionAI/Data/Dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_filepath = 'NumRecognitionAI/Data/Dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = 'NumRecognitionAI/Data/Dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    #
    # Load MNIST dataset
    #
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (img_train, label_train), (img_test, label_test) = mnist_dataloader.load_data()

    # print("size of Image: ", len(img_test[0]))
    # print("size of Image: ", len(img_test[0][0]))

    # create network
    # 2 layers not including input or output
    # size 20 layers
    # 28 x 28 images where each input is a pixel (784 inputs)
    # outputs 0 through 9 (10 outputs)
    network = Network(3, 20, 784, 10)

    while True:
        # infinite loop to test and train network
        user_in = input("Choose Function:\n"
                        "1) train\n"
                        "2) test\n"
                        "3) exit\n")

        match user_in:
            case 1:
                train(network, img_train, label_train)
            case 2:
                test(network, img_test, label_test)
            case default:
                exit(0)
