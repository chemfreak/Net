# # # imports # # # 

import argparse
import itertools
import os
import sys

from Net import *

from matplotlib import pyplot as plt

# # # Global Variables # # #


description = "Digit recognition"

identity = "Klösch, Christoph 01125514"


# # # Classes # # #


# # # Functions # # #

def extract_data(filename, path):
    """
    extract trainings data from folder
    """
    folder_name = data_folder + "/" + path

    print(folder_name)

    samples = random.sample(os.listdir(
        folder_name), 100

    )

    image_list = [(int(filename[1]), plt.imread(folder_name + "/" + img)[:, :,
                                     0]) for
                  img in
                  samples]

    return image_list


def filter_gray(img):
    """
    Filter out gray values
    """

    img = img
    for i, pi in enumerate(img):
        for j, pj in enumerate(pi):
            if 0 < pj < 1:
                img[i, j] = 0
    return img


# # # Parser # # #

def parse_args():
    """parse command line arguments and return arguments in list"""

    # create parser object
    parser = argparse.ArgumentParser(description=description)

    # add arguments
    parser.add_argument('--identity',
                        action='store_true',
                        help='[Optional] Print identity of student and '
                             'exit: Lastname, Firstname Matrikelnumber',
                        default=False
                        )

    parser.add_argument('-i', '--infile',
                        action='store',
                        help='Specify a .sif interaction file name.',
                        type=str
                        )

    return parser.parse_args()


# # # Main Function # # #

if __name__ == "__main__":

    # parse command line arguments
    args = parse_args()

    # if '--identity' is used, show identity and exit
    if args.identity:
        print(identity)
        sys.exit()


    # get data from folder
    data_folder = "./by_class/by_class"
    print("Extracting data from folders:")

    data = [extract_data(filename, filename + "/train_3" + filename[1]) for
            filename in os.listdir(
            data_folder) if
            "n" in filename]

    # collapse into 1D array
    data = list(itertools.chain.from_iterable(data))

    # initialize network
    net = Net([LastLayer(16384, 10)])

    # fit training data
    train_losses, eval_losses = net.fit(data)

    # plot losses
    plt.plot(range(len(train_losses)), train_losses, label="train_losses")
    plt.plot(range(len(eval_losses)), eval_losses, label="eval_losses")

    plt.legend()
    plt.show()

    # ask for image to read
    while True:

        try:

            # prompt
            file = input("What file do you wanna read?")

            # read
            img = plt.imread(file)[:, :, 0]

            # Training data only consists of black and white,
            # while self-made digits can contain grayscale
            img = filter_gray(img)

            # predict
            predict, out = net.predict(img)

            print("The digit shown is a", predict)
            print("output:", out)

        except FileNotFoundError as e:
            print("File not found!")
