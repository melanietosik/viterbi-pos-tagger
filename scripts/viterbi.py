# Melanie Tosik
# NLP, Viterbi part-of-speech (POS) tagger

from __future__ import division

import math


class Viterbi():
    """
    Viterbi
    https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, O, S, Y, A, B):

        self.O = O  # Observation space
        self.S = S  # State space
        self.Y = Y  # Sequence of observations
        self.A = A  # Transition matrix
        self.B = B  # Emission matrix

        self.N = len(self.O)
        self.K = len(self.S)

        # Word index lookup table
        self.lookup = {}
        for i, word in enumerate(self.O):
            self.lookup[word] = i

        self.T = len(Y)
        self.T1 = [[0] * self.T for i in range(self.K)]
        self.T2 = [[None] * self.T for i in range(self.K)]

        # Predicted tags
        self.X = [None] * self.T

    def decode(self):
        """
        Run the algorithm
        """
        # Initialize start probabilities
        self.init()
        # Forward step
        self.forward()
        # Backward step
        self.backward()
        return self.X

    def init(self):
        """
        Initialize start probabilities
        """
        s_idx = self.S.index("--s--")
        for i in range(self.K):
            if self.A[s_idx][i] == 0:
                self.T1[i][0] = float("-inf")
                self.T2[i][0] = 0
            else:
                self.T1[i][0] = \
                    math.log(self.A[s_idx][i]) + \
                    math.log(self.B[i][self.lookup[self.Y[0]]])
                self.T2[i][0] = 0

    def forward(self):
        """
        Forward step
        """
        for i in range(1, self.T):

            if i % 5000 == 0:
                print("Words processed: {:>8}".format(i))

            for j in range(self.K):

                best_prob = float("-inf")
                best_path = None

                for k in range(self.K):

                    prob = self.T1[k][i - 1] + \
                        math.log(self.A[k][j]) + \
                        math.log(self.B[j][self.lookup[self.Y[i]]])

                    if prob > best_prob:
                        best_prob = prob
                        best_path = k

                self.T1[j][i] = best_prob
                self.T2[j][i] = best_path

    def backward(self):
        """
        Backward step
        """
        z = [None] * self.T
        argmax = self.T1[0][self.T - 1]

        for k in range(1, self.K):
            if self.T1[k][self.T - 1] > argmax:
                argmax = self.T1[k][self.T - 1]
                z[self.T - 1] = k

        self.X[self.T - 1] = self.S[z[self.T - 1]]

        for i in range(self.T - 1, 0, -1):
            z[i - 1] = self.T2[z[i]][i]
            self.X[i - 1] = self.S[z[i - 1]]
