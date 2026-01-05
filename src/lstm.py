import math
import random

from utils import *


class LongShortTermRNN:

    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Input Gate
        self.Wi, self.Ui, self.bi = generate_random_gate_weights(hidden_size, input_size)
        # Forget Gate
        self.Wf, self.Uf, self.bf = generate_random_gate_weights(hidden_size, input_size)
        # Output Gate
        self.Wo, self.Uo, self.bo = generate_random_gate_weights(hidden_size, input_size)
        # Cell Candidate
        self.Wc, self.Uc, self.bc = generate_random_gate_weights(hidden_size, input_size)

        # Output layer
        self.V = [
            [random.random() for _ in range(output_size)] for _ in range(hidden_size)
        ]
        self.by = [random.random() for _ in range(output_size)]

    def forward(self, inputs):
        h = [0] * self.hidden_size
        c = [0] * self.hidden_size

        for x in inputs:
            i_gate = [
                sigmoid(
                    sum(x[j] * self.Wi[j][k] for j in range(self.input_size))
                    + sum(h[j] * self.Ui[j][k] for j in range(self.hidden_size))
                    + self.bi[k]
                )
                for k in range(self.hidden_size)
            ]
            f_gate = [
                sigmoid(
                    sum(x[j] * self.Wf[j][k] for j in range(self.input_size))
                    + sum(h[j] * self.Uf[j][k] for j in range(self.hidden_size))
                    + self.bf[k]
                )
                for k in range(self.hidden_size)
            ]
            o_gate = [
                sigmoid(
                    sum(x[j] * self.Wo[j][k] for j in range(self.input_size))
                    + sum(h[j] * self.Uo[j][k] for j in range(self.hidden_size))
                    + self.bo[k]
                )
                for k in range(self.hidden_size)
            ]
            c_candidate = [
                tanh(
                    sum(x[j] * self.Wc[j][k] for j in range(self.input_size))
                    + sum(h[j] * self.Uc[j][k] for j in range(self.hidden_size))
                    + self.bc[k]
                )
                for k in range(self.hidden_size)
            ]

            c = [
                f_gate[k] * c[k] + i_gate[k] * c_candidate[k]
                for k in range(self.hidden_size)
            ]
            h = [o_gate[k] * tanh(c[k]) for k in range(self.hidden_size)]

        y = [
            sum(h[j] * self.V[j][k] for j in range(self.hidden_size)) + self.by[k]
            for k in range(self.output_size)
        ]

        # softmax for next-word probabilities
        exp_y = [math.exp(v) for v in y]
        total = sum(exp_y)
        y = [v / total for v in exp_y]

        self.last_h = h
        self.last_c = c
        self.last_inputs = inputs
        self.last_y = y
        return y

    def train(self, inputs, target_index):
        output = self.forward(inputs)
        target = [0] * self.output_size
        target[target_index] = 1
        error = [target[i] - output[i] for i in range(self.output_size)]

        # update output layer
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.V[j][k] += self.lr * error[k] * self.last_h[j]
        for k in range(self.output_size):
            self.by[k] += self.lr * error[k]
