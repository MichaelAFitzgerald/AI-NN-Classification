import random
import math
import pandas as pd
import sys


# Class that is the perceptron, initialized to values in the Perl slides with modifications made based on documentation
class Perceptron:
    # initializes
    cycles = 5000
    input_num = 2
    alpha = 0.3
    weights = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 1]
    totalError = 0
    activation = 'Hard'

    # requires a definition of activation before running
    def __init__(self, activation):
        self.activation = activation

    # activation function, modified for unipolar
    def signal(self, net, activation='Hard'):
        if self.activation == 'Soft':
            k = 0.5  # gain
            x = net
            r = 1 / (1 + math.exp(-k * x))
            return r
        elif self.activation == 'Hard':
            x = net
            if x > 0:
                y = 1
            else:
                y = 0
            return y
        else:
            raise ValueError('Incorrect Input')

    # training function as implemented in slides, with slight modification to define pattern as input
    def train(self, error_lim, training_data):
        for i in range(self.cycles):
            self.totalError = 0
            for j in range(len(training_data)):
                net = 0
                for k in range(self.input_num):
                    if k == 0:
                        net = net + self.weights[k] * training_data['Height'][j]
                    else:
                        net = net + self.weights[k] * training_data['Weight'][j]
                net = self.weights[2] + net
                output = self.signal(net)
                error = training_data['Gender'][j] - output
                self.totalError = self.totalError + (error ** 2)
                learn = self.alpha * error
                self.print_data(i, j, net, error, learn)
                for z in range(len(self.weights)):
                    if z == 0:
                        self.weights[z] = self.weights[z] + learn * training_data['Height'][j]
                    elif z == 1:
                        self.weights[z] = self.weights[z] + learn * training_data['Weight'][j]
                    else:
                        self.weights[z] = self.weights[z] + learn
            print("Total Error: {0:1.10f}".format(self.totalError))
            if self.totalError < error_lim:
                break
        return self.weights

    # testing function to test a weight pattern
    def test(self, weights, testing_data):
        trueMale = 0
        trueFemale = 0
        falseMale = 0
        falseFemale = 0
        for i in range(0, len(testing_data)):
            pred = weights[0] * testing_data['Height'][i] + weights[1] * testing_data['Weight'][i] + weights[2]
            if pred > 0 and testing_data['Gender'][i] > 0:
                trueFemale = trueFemale + 1
            elif pred <= 0 and testing_data['Gender'][i] <= 0:
                trueMale = trueMale + 1
            elif pred > 0 and testing_data['Gender'][i] <= 0:
                falseMale = falseMale + 1
            else:
                falseFemale = falseFemale + 1
        print("trueMale: {0} trueFemale: {1} falseMale: {2} falseFemale: {3}".format(trueMale, trueFemale, falseMale,
                                                                                     falseFemale))
        accuracy = (trueMale + trueFemale) / (trueMale + trueFemale + falseMale + falseFemale)
        print("Accuracy: {0}".format(accuracy))

    # easy-access print data function
    def print_data(self, ite, p, net, err, lrn):
        print("ite={0:3} p={1} net={2:5.2f} err ={3:6.3f} lrn ={4:6.3f} wei: {5}".format(ite, p, net, err, lrn,
                                                                                         str(self.weights)))
