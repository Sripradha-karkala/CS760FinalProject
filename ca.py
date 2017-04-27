import numpy as np
import random
import csv
import sys
import math
import argparse
import matplotlib.pyplot as plt

a = np.array([1.,2.,3.])
b = np.array([2.,4.,6.])

data = np.array([a,b])

class DataType:
    """There are two kinds of CA state data - continuous-valued and integer-valued.
    Extend this if you want to add additional options.
    """
    DATA_WITH_FLOATS = 1
    DATA_WITH_INTS = 2

class Data:
    """Loads data from files. Represents time-series and graph data from a CA simulation.

    Attributes:
    partitions - A list of matrices representing slices of data.
    """

    """Constructor: Given a CSV file of flu rates, create a list of matrices for training and testing.

    Arguments:
    file_name -- CSV flu rates file
    neighbor_file -- File containing cities and weights
    num_folds -- number of folds to partition

    Keyword arguments (kwargs):
    num_folds -- If this is provided, the data will be divided into cross validation folds.
    split -- If this is provided, then this proportion of data will be put in the first partition.
    """
    def __init__(self, file_name, neighbor_file, data_type, **kwargs):

        with open(file_name) as input_file:
            csv.reader(input_file, delimiter = ',')
            data = list(input_file)
        data = [numbers.split(',') for numbers in data]

        #Parse the neighbour file to create the weighted graph
        neighbor_data = []
        with open(neighbor_file) as f_neighbor:
            #csv.reader(f_neighbor, delimiter=',')
            lines = f_neighbor.readlines();
            for line in lines:
                row = line.split(',')
                neighbor_data.append(row);
        #neighbor_data
        self.cities = []
        for city in neighbor_data:
            self.cities.append(city[0]) #The first city is the source

        # Build the graph based on adjacency matrix
        # 0 if no edge exists between two cities
        self.graph = [[0 for i in self.cities] for j in self.cities]
        for row in neighbor_data:
            for i in range(1,len(row)-2,2):
                self.graph[self.cities.index(row[0])][self.cities.index(row[i])] = int(row[i+1])

        # Slice the data into training and test sets, or into folds.
        if data_type == DataType.DATA_WITH_FLOATS:
            d = np.array(data)[1:,1:].astype(float) # first column has date. Why are we also getting rid of the first row?
        else:
            d = np.array(data)[1:,1:].astype(int) # first column has date

        if 'num_folds' in kwargs and 'split' in kwargs:
            raise ValueError('Both num_folds and split were passed as arguments, so the slicing method is ambiguous.')

        self.partitions = []
        if 'num_folds' in kwargs:
            # Slice data into n folds
            num_folds = kwargs['num_folds']
            partition_size = int(math.ceil(len(d)/num_folds))
            for i in range(num_folds):
                self.partitions.append(d[(i*partition_size):((i+1)*partition_size),:])
        elif 'split' in kwargs:
            # Slice data in train and test sets
            split = kwargs['split']
            trainset_size = int(math.ceil((1.0 - split) * len(d)))
            self.partitions.append(d[:trainset_size,:])
            self.partitions.append(d[trainset_size:,:])

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
        return magnitude * np.random.rand(self.neighborhood_size + 2) - float(magnitude) / 2 # +1 for self, +1 for bias

    def mutate(self, amount):
        offset = self.make_weights(amount)
        return UpdateRule(self.neighborhood_size, self.weights + offset)

    def perturb_single_weight(self, amount):
        # TODO
        # Choose a random index and mutate it
        index = random.randint(0, len(self.weights) - 1)



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

"""Calculate the error of a given CA node on a set of training data and plot values
Arguments:
rule -- an instance of UpdateRule
data -- a matrix of CA values, where data[t] is an array of all cells values at time t
city_index -- index of the node in question"""
def plot_error(rule, data, city_index):
    ca = CellularAutomaton(data[0], rule)
    output = []
    desired_output = []
    error = []
    # Calculate output and error for each time step
    for t in range(1, len(data)):
        ca.update()
        current_output = ca.get_values()[city_index]
        current_desired_output = data[t][city_index]
        current_error = abs(current_output - current_desired_output)
        output.append(current_output)
        desired_output.append(current_desired_output)
        error.append(current_error)

    # Plot output, desired output, and error
    plt.plot(output, label='output')
    plt.plot(desired_output, label='desired')
    plt.plot(error, label='error')
    plt.legend()
    plt.xlabel('Iteration')
    plt.show()

def main(args):
    # Create and run a CA.
    data = Data(args.input_file, args.neighbor_file, args.split, data_type = DATA_WITH_FLOATS)
    rule = UpdateRule(args.neighbor)
    evaluate_rule(rule, data.partitions[0], args.batch)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type = str,
            help='Specify the path/filename of the input data.')
    parser.add_argument('neighbor_file', type = str,
            help='Specify the path/filename of the neighbor data.')
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