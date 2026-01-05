import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return math.tanh(x)


def dsigmoid(y):
    return y * (1 - y)


def dtanh(y):
    return 1 - y**2


def one_hot(index, size):
    vec = [0] * size
    vec[index] = 1
    return vec


def generate_random_gate_weights(hidden_size, input_size):
    return (
        [[random.random() for _ in range(hidden_size)] for _ in range(input_size)],
        [[random.random() for _ in range(hidden_size)] for _ in range(hidden_size)],
        [random.random() for _ in range(hidden_size)],
    )
