import numpy as np
import random
import csv
import sys
import math
import argparse

a = np.array([1.,2.,3.])
b = np.array([2.,4.,6.])

data = np.array([a,b])

# TODO refactor to a lists of training and test matrices, by batch_size
class Data:
    """Given a CSV file of flu rates, create two matrices for training and testing.
    These can be accessed by the properties train_data and test_data.

    Arguments:
    file_name -- CSV flu rates file
    testset_split -- proportion of data to use for testing
    """
    def __init__(self, file_name, testset_split):
        with open(file_name) as input_file:
            csv.reader(input_file, delimiter = ',')
            data = list(input_file)
        data = data[12:] # first 12 lines have comments
        data = [numbers.split(',') for numbers in data]
        d = np.array(data)[:,1:].astype(int) # first column has date
        trainset_size = int(math.ceil((1.0 - testset_split) * len(d)))
        self.train_data = d[:trainset_size,:]
        self.test_data = d[trainset_size:,:]

class UpdateRule:
    """Constructs an update function, which can be called like any normal Python function.
    This represents a point in the parameter space of the CA.

    Arguments:
    [TODO remove] neighborhood_size -- the number of neighbors of a cell
    weights (optional) -- an weight matrix
    """
    def __init__(self, neighborhood_size, weights = None):
        self.neighborhood_size = neighborhood_size
        if weights is None:
            self.weights = self.make_weights(1)
        else:
            self.weights = weights

    def __call__(self, cell_value, neighbor_values):
        return self.weights[0] * cell_value \
            + self.weights[1:-1].dot(neighbor_values) + self.weights[-1]

    def make_weights(self, magnitude):
        return magnitude * np.random.rand(self.neighborhood_size + 2) # +1 for self, +1 for bias

    def perturb(self, amount):
        offset = self.make_weights(amount)
        return UpdateRule(self.neighborhood_size, self.weights + offset)


class CellularAutomaton:
    """Given an update rule an initial values, the CA can repeatedly update its own state.

    Arguments:
    initial_values -- an array of cell values at time 0
    update_rule -- an instance of UpdateRule to update the cell values"""
    def __init__(self, initial_values, update_rule):
        self.values = np.copy(initial_values)
        self.update_rule = update_rule

    def get_neighbors(self, i):
        return np.append(self.values[:i], self.values[i+1:])

    def update(self):
        for i in range(len(self.values)):
            neighbors = self.get_neighbors(i)
            self.values[i] = self.update_rule(self.values[i], neighbors)

    def get_values(self):
        return self.values

"""Calculate the error of a given CA on a set of training data
Arguments:
rule -- an instance of UpdateRule
data -- a matrix of CA values, where data[t] is an array of all cells values at time t"""
def evaluate_rule(rule, data):
    ca = CellularAutomaton(data[0], rule)
    error = 0.0
    debug_errors = []
    for t in range(1, len(data)):
        ca.update()
        error += sum(abs(ca.get_values() - data[t]))
        debug_errors.append(error)
    # print 'cumulative errors: %s' % debug_errors
    return error

def main(args):
    data = Data(args.input_file, args.split)
    rule = UpdateRule(args.neighbor)
    evaluate_rule(rule, data.train_data, args.batch)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type = str,
            help='Specify the path/filename of the input data.')
    parser.add_argument('-s', '--split', type = float, default = 0.3,
            help = 'Specify the portion of data to use as testset, e.g. 0.3.')
    parser.add_argument('-n', '--neighbor', type = int, default = 2,
            help = 'Specify the number of neighbors to use.')
    parser.add_argument('-b', '--batch', type = int, default = 2,
            help = 'Specify the size of the data window in a batch.')
    return parser

def parse_args():
    parser = make_argument_parser()
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())

