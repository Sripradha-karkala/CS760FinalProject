"""Some hand-crafted CA's to test on
"""

from ca import make_argument_parser as make_ca_argument_parser, Data, UpdateRule, CellularAutomaton, plot_error
import random
import numpy as np

N_EXAMPLES = 2

def make_example_0(neighbor):
    """ Constant example """
    dimension = 2
    n_rows = dimension
    n_cols = 2 * dimension + 1
    weights = np.zeros((n_rows, n_cols))

    # Columns 0 and 1 are weights for the self-cell
    weights[0, 0] = weights[1, 1] = 1
    weights[1, -1] = 10
    return UpdateRule(neighbor, weights)

def make_example_1(neighbor):
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
    return UpdateRule(neighbor, weights)

def run_example(args):
    example = args.example
    if example > N_EXAMPLES - 1:
        raise ValueError('There is no example number %s' % example)

    # Load data
    data = Data.create_from_args(args)

    # Create the update rule for this example
    if example == 0:
        rule = make_example_0(args.neighbor)
    elif example == 1:
        rule = make_example_1(args.neighbor)

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
