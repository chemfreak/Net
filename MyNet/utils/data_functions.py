import random


def split_data(data, eval_fraction=0.1):
    """
    split data into training and validation set
    :param eval_fraction: fraction of the data to be used for evaluation
    :return: train_data, eval_data
    """

    # shuffle data
    random.shuffle(data)

    # len of evaluation data
    len_eval = int(len(data) * eval_fraction)

    train_data = data[:-len_eval]
    eval_data = data[-len_eval:]

    return train_data, eval_data