### word2vec configuration ###


def embedding_size():
    return 300


def epoch_num():
    return 2


def batch_size():
    return 10


def window_size():
    return 5


def learning_rate():
    return 0.01


def neg_num():
    return 5


def method():
    return 0  # 0 : negative sampling, 1 : hierarchical softmax


def subsampling():
    return False  # True : use subsampling, False : not use


def skip_prob():  # 0.6 : 60% of the batch used in training
    return 0.6
