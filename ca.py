import numpy as np
import random
import csv
import sys
import math
import argparse
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

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

    @staticmethod
    def create_from_args(args):
        """Factory method for convenience. Pass it the args object and it return a Data object."""
        if 'split' in args:
            return Data(args.input_file, args.neighbor_file, DataType.DATA_WITH_FLOATS, split=args.split)
        else:
            return Data(args.input_file, args.neighbor_file, DataType.DATA_WITH_FLOATS, num_folds=args.num_folds)

    """Constructor: Given a CSV file of flu rates, create a list of matrices for training and testing.
    Arguments:
    file_name -- CSV flu rates file
    neighbor_file -- File containing cities and weights
    num_folds -- number of folds to partition
    Keyword arguments (kwargs):
    num_folds -- If this is provided, the data will be divided into cross validation folds.
    split -- If this is provided, then this proportion of data will be put in the first partition.
    """
    def __init__(self, file_name, neighbor_file, data_type, num_folds = None, split = None):

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
        self.graph = np.zeros((len(self.cities), len(self.cities)))
        for row in neighbor_data:
            for i in range(1,len(row)-2,2):
                self.graph[self.cities.index(row[0])][self.cities.index(row[i])] = int(row[i+1])

        # Process the data list into a numpy array
        if data_type == DataType.DATA_WITH_FLOATS:
            d = np.array(data)[1:,1:].astype(float) # first column has date. Why are we also getting rid of the first row?
        else:
            d = np.array(data)[1:,1:].astype(int) # first column has date

        # Slice the data into training and test sets, or into folds.
        if num_folds is not None and split is not None:
            raise ValueError('Both num_folds and split were passed as arguments, so the slicing method is ambiguous.')

        self.partitions = []
        if num_folds is not None:

            # Slice data into n folds
            partition_size = int(math.ceil(len(d)/num_folds))
            for i in range(num_folds):
                fold = d[(i*partition_size):((i+1)*partition_size), :]
                self.partitions.append(fold)
        elif split is not None:
            # Slice data in train and test sets
            trainset_size = int(math.ceil((1.0 - split) * len(d)))
            self.partitions.append(d[:trainset_size,:])
            self.partitions.append(d[trainset_size:,:])

class UpdateRule:
    """An update function can be called like any normal Python function.
    This represents a point in the parameter space of the CA.
    Attributes:
    weights -- a numpy array with d rows and (2d + 1) columns, where d is the dimension of a cell value.
                The first d columns are weights for the cell, the next d are weights for the neighbors, and the last one is bias.
    dimension -- d; each cell value is a d-length vector.
    """

    WEIGHT_RANGE = 2 # values are in the interval -2 to 2
    dimension = 2

    def __init__(self, graph, weights = None):
        """Initialize a random update rule, or pass an update rule matrix.
        Arguments:
        graph: Graph of cities with weights
        weights (optional) -- an weight matrix
        """
        if weights is None:
            self.weights = self.make_weights()
        else:
            self.weights = weights
        self.graph = graph

    def __call__(self, cell_index, cell_value, neighbor_indices, neighbor_values):
        """Calculate f(x, N(x)).
        """
        z = self.get_z(cell_index, neighbor_indices, neighbor_values)

        # print 'BEGIN'
        # print self.weights[:, 0:self.dimension].dot(cell_value)
        # print self.weights[:, self.dimension:2*self.dimension], z
        # print self.weights[:, self.dimension:2*self.dimension].dot(z)
        # print self.weights[:, 2*self.dimension:].flatten()
        # print 'END'

        # self, neighbors, bias
        return self.weights[:, 0:self.dimension].dot(cell_value) \
            + self.weights[:, self.dimension:2*self.dimension].dot(z) \
            + self.weights[:, 2*self.dimension:].flatten()

    def get_z(self, cell_index, neighbor_indices, neighbor_values):
        """Return the weighted sum of the neighbor values.
        cell_index -- the index of the target cell
        neighbor_indices -- a list of indices of source cells
        neighbor_values -- a list of 2x1 numpy matrices, or an empty list"""

        # All the cells which are non-zero are basically neighbours
        z = np.zeros(2)
        for neighbor_index, neighbor_value in zip(neighbor_indices, neighbor_values):
            weight = self.graph[neighbor_index, cell_index]
            z += weight * neighbor_value

        if len(neighbor_indices) != 0:
            for i in range(len(z)):
                z[i] /= float(len(neighbor_indices))
        return z

    def make_weights(self):
        """Create a weights matrix uniformly distributed over -WEIGHT_RANGE, WEIGHT_RANGE."""
        n_rows = self.dimension
        n_cols = 2 * self.dimension + 1
        return 2 * self.WEIGHT_RANGE * (np.random.rand(n_rows, n_cols) - 0.5)

    def crossover(self, update_rule):
        """Create a new update rule which is blended between this one and the one passed."""
        # [TODO] Allow not all variables to be blended
        child_weights = np.zeros(self.weights.shape)

        # Work with 1d arrays for generality
        parent_a_flat = self.weights.flat
        parent_b_flat = update_rule.weights.flat
        child_flat = child_weights.flat

        for i in range(len(child_flat)):
            blend = random.random()
            child_flat[i] = blend * parent_a_flat[i] + (1 - blend) * parent_b_flat[i]
        return UpdateRule(self.graph, child_weights)

    def mutate(self, rate):
        """Change a given fraction of the weights to new random values."""
        weights = self.weights.flat
        for i in range(len(weights)):
            if random.random() < rate:
                weights[i] = 2 * self.WEIGHT_RANGE * (random.random() - 0.5)

class CellularAutomaton:
    """Given an update rule an initial cell values, the CA can repeatedly update its own state.
    Attributes:
    cells -- a numpy array where cells[i][j] represents the jth dimension of the ith cell.
    """

    # TODO - don't define this twice in UpdateRule
    dimension = 2

    def __init__(self, initial_values, update_rule):
        """Initialze a CA. The first dimension will be set to the initial_values, and all other dimensions are set to 0.
        Arguments:
        initial_values -- an array of cell values at time 0
        update_rule -- an instance of UpdateRule to update the cell values"""
        self.cells = np.ones((len(initial_values), self.dimension))
        self.cells[:, 0] = np.copy(initial_values)
        self.update_rule = update_rule

    def get_neighbors(self, i):
        """Return a list of indices of the cells with edges to the cell index i"""
        # Assume 1D grid of cells.
        neighbors = []
        graph = self.update_rule.graph
        for index in range(len(graph[0])):  # Graph is an adj. matrix, so length will be same
            if graph[i][index] != 0: # is neighbour
                neighbors.append(index)
        return neighbors

    def update(self):
        for i in range(len(self.cells)):
            neighbors = self.get_neighbors(i)
            #print neighbors
            #print len(self.cells)
            # print self.cells[neighbors]
            self.cells[i] = self.update_rule(i, self.cells[i], neighbors, self.cells[neighbors])

    def get_values(self):
        return self.cells

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
        actual = ca.get_values()[:, 0].flatten()
        desired = data[t]
        error += sum(abs(actual - desired)) # TODO use squared error
        debug_errors.append(error)
    # print 'cumulative errors: %s' % debug_errors
    return error

"""Generates output from a CA on a dataset
Arguments:
rule -- an instance of UpdateRule
data -- a matrix of CA values, where data[t] is an array of all cells values at time t"""
def generate_output(rule, data):
    ca = CellularAutomaton(data[0], rule)
    output = []
    output.append(data[0])
    for t in range(1, len(data)):
        ca.update()
        output.append(ca.get_values()[:, 0].flatten())
    output = np.asarray(output)
    return output

"""Calculates average pearson correlation across all cities between output values
and true values in the dataset
Arguments:
rule -- an instance of UpdateRule
data -- a matrix of CA values, where data[t] is an array of all cells values at time t"""
def pearson_correlation(rule, data):
    output = generate_output(rule, data)
    data = np.asarray(data)
    num_cities = data.shape[1]
    city_correlations = []
    for i in range(num_cities):
        correlation = pearsonr(data[:,i], output[:,i])
        print(correlation[0])
        city_correlations.append(correlation[0])
    return np.mean(city_correlations)


"""Calculate the error of a given CA node on a set of training data and plot values
Arguments:
rule -- an instance of UpdateRule
data -- a matrix of CA values, where data[t] is an array of all cells values at time t
city_index -- index of the node in question"""
def plot_error(rule, data, city_index):
    print('Final weights:\n%s' % rule.weights)
    ca = CellularAutomaton(data[0], rule)
    outputs = []
    desired_output = []
    error = []
    # Calculate outputs and error for each time step
    for t in range(1, len(data)):
        ca.update()
        current_output = np.copy(ca.get_values()[city_index])
        current_desired_output = data[t][city_index]
        current_error = abs(current_output[0] - current_desired_output)
        outputs.append(current_output)
        desired_output.append(current_desired_output)
        error.append(current_error)

    # Plot output, desired output, and error
    for i in range(len(outputs[0])):
        output = [o[i] for o in outputs]
        plt.plot(output, label='output_%s' % i)
    plt.plot(desired_output, label='desired')
    plt.plot(error, label='error')
    plt.legend()
    plt.xlabel('Iteration')
    plt.show()

def main(args):
    # Create and run a CA.
    data = Data.create_from_args(args)
    rule = UpdateRule(data.graph)
    evaluate_rule(rule, data.partitions[0])

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type = str,
            help='Specify the path/filename of the input data.')
    parser.add_argument('neighbor_file', type = str,
            help='Specify the path/filename of the neighbor data.')

    training_group = parser.add_mutually_exclusive_group()
    training_group.add_argument('-s', '--split', type = float, default = 0.3,
            help = 'Split data into training and test set, and use the specified proportion of data for the training set.')
    training_group.add_argument('-f', '--num_folds', type = int,
            help = 'Split data into multiple folds and train using cross validation.')

    parser.add_argument('-n', '--neighbor', type = int, default = 2,
            help = 'Specify the number of neighbors to use.')
    return parser

def parse_args():
    parser = make_argument_parser()
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())