#  Main file for project
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
    print("Loading MNIST DATABASE\n")
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (img_train, label_train), (img_test, label_test) = mnist_dataloader.load_data()
    print("Finished loading MNIST DATABASE\n")

    # print("size of Image: ", len(img_test[0]))
    # print("size of Image: ", len(img_test[0][0]))
    # print("first label: ", label_test[0])

    # create network
    # 2 layers not including input or output
    # size 20 layers
    # 28 x 28 images where each input is a pixel (784 inputs)
    # outputs 0 through 9 (10 outputs)
    num_layers, layer_size, num_inputs, num_outputs = 3, 100, 784, 10

    # allow user input on network parameters
    while True:
        user_in = input("Select Network Parameters\n"
                        "Leave as default to use supplied CSV\n"
                        "1) num_layers - Current: {}\n"
                        "2) layer_size - Current: {}\n"
                        "3) continue\n".format(num_layers, layer_size))
        match int(user_in):
            case 1:
                num_layers = int(input("Enter Integer Value\n"))
            case 2:
                layer_size = int(input("Enter Integer Value\n"))
            case default:
                break

    print("Initializing Network with:\n"
          "{} Hidden Layers\n"
          "{} Layers Size\n"
          "{} Inputs\n"
          "{} Outputs\n".format(num_layers, layer_size, num_inputs, num_outputs))

    # default and weight range, can be changed later
    network = Network(num_layers, layer_size, num_inputs, num_outputs, 1.0, 10.0)

    while True:
        # infinite loop to test and train network
        user_in = input("Choose Function:\n"
                        "1) train\n"
                        "2) test\n"
                        "3) exit\n")

        match int(user_in):
            case 1:
                train(network, img_train, label_train)
            case 2:
                test(network, img_test, label_test)
            case default:
                exit(0)
