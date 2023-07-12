# TODO: add categorize function n -> vector of n with 1 at 1, default =10
import random


def shift_image(img, shift=30):
    # To-do: zahlen random verschieben
    random_num_1 = random.randint(0, shift)
    random_num_2 = random.randint(0, shift)
    l = img.shape[0]
    img = img[random_num_1:(l - shift) + random_num_1,
          random_num_2:(l - shift) + random_num_2]

    return img