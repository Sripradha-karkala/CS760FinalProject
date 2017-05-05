"""Some hand-crafted CA's to test on
How to add another example:
1. Write make_example_k
2. Increment N_EXAMPLES
3. Add a conditional for your example k
"""

from ca import make_argument_parser as make_ca_argument_parser, Data, UpdateRule, CellularAutomaton, plot_error
import random
import numpy as np

N_EXAMPLES = 4

def make_example_0(graph):
    """ Constant example """
    dimension = 2
    n_rows = dimension
    n_cols = 2 * dimension + 1
    weights = np.zeros((n_rows, n_cols))

    # Columns 0 and 1 are weights for the self-cell
    weights[0, 0] = weights[1, 1] = 1
    weights[1, -1] = 10
    return UpdateRule(graph, weights)

def make_example_1(graph):
    """ Sample oscillator """
    dimension = 2
    n_rows = dimension
    n_cols = 2 * dimension + 1
    weights = np.zeros((n_rows, n_cols))

    # Rule: weights[i, j] is influence of dimension j on dimension i

    # Columns 0 and 1 are weights for the self-cell
    weights[0, 0] = weights[1,1] = 1 # exponential decay
    weights[1, 0] = 0.03
    weights[0, 1] = -0.03
    weights[0, -1] = 10
    weights[1, -1] = -100
    return UpdateRule(graph, weights)

def make_example_2(graph):
    # Result of genetic algorithm learning, 1000 in population, for 400 generations.
    # Shows simple convergence to the mean.
    weights = np.array([
        [ 0.61937729, 0.39776793, -0.05319007, 0.17862905, 1.96754799],
        [-0.3371428, -0.05163913, 0.92034044, 0.2732566, 1.98554387]
    ])
    return UpdateRule(graph, weights)

def make_example_3(graph):
    weights = np.array([
        # [1.42401506e+00, -4.97480473e-01, 1.92521117e-04, -1.84918083e-04, 1.20461306e+02],
        # [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
        [1.2, -1, 0, 0, 0],
        [1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]
    ])
    return UpdateRule(graph, weights)

def run_example(args):
    example = args.example
    if example > N_EXAMPLES - 1:
        raise ValueError('There is no example number %s' % example)

    # Load data
    data = Data.create_from_args(args)

    # Create the update rule for this example
    if example == 0:
        rule = make_example_0(data.graph)
    elif example == 1:
        rule = make_example_1(data.graph)
    elif example == 2:
        rule = make_example_2(data.graph)
    elif example == 3:
        rule = make_example_3(data.graph)
    else:
        raise ValueError('Conditional for example %s is not written' % example)

    # Plot the results
    plot_error(rule, data.partitions[0], 0)
    plot_error(rule, data.partitions[0], 1)
    plot_error(rule, data.partitions[0], 2)


def make_argument_parser():
    parser = make_ca_argument_parser()
    parser.add_argument('-e', '--example', type = int, default = 0,
            help = 'Specify the index of the example CA you wish to run, from %s to %s.' % (0, N_EXAMPLES - 1))
    return parser

def parse_args():
    parser = make_argument_parser()
    return parser.parse_args()

if __name__ == '__main__':
    # Create a genetic trainer
    args = parse_args()
    run_example(args)

    trainer = GeneticTrainer(args)
    data = Data.create_from_args(args)

    if args.num_folds is None:
        # Train it with a train-test split.
        basic_train(trainer, data)
    else:
        # Train it with CV trainer if -f flag is passed
        cross_validation_train(trainer, data)
