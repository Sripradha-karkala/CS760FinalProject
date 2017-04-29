from ca import make_argument_parser as make_ca_argument_parser, Data, UpdateRule, CellularAutomaton, evaluate_rule, DataType
from trainer import Trainer, basic_train, cross_validation_train
import random
import numpy as np
import matplotlib.pyplot as plt

NUM_POPULATION = 100
NUM_KEPT = 20 # keep this many for the next generation. must be even.
MUTATION_RATE = 0.2 # the proportion of chromosomes that are mutated
MUTATION_DISTANCE = 0.01

HUGE_NUMBER = 1e20

def random_lerp(a, b):
    """Computes a new value between a and b.

    Arguments:
    a -- a float
    b -- a float
    """
    rand = random.random()
    return rand * a + (1 - rand) * b

def make_ga_argument_parser():
    parser = make_ca_argument_parser()
    parser.add_argument('-g', '--generations', type = int, default = 5,
            help = 'Specify the number of generations to train for.')
    parser.add_argument('-m', '--mutate', type = float, default = 0.1,
            help = 'Specify the amount to mutate the weights by each generation.')
    return parser

def get_min_k(k, objects, scores):
    """Return the k objects with the lowest scores.

    Arguments:
    objects - a list
    scores - a list of numbers
    """
    # Believe in the one-liner!
    return [y[1] for y in sorted(zip(scores, objects), key=lambda x: x[0])][0:k]

def plot_timeseries(values):
    values_1 = [v[0] for v in values]
    # plt.ylim(min(values_1), max(values_1))
    # plt.plot(range(len(values)), values_1)
    plt.plot(values_1)
    plt.ylabel('Error')
    plt.show()

def make_random_rule(neighborhood_size):
    return UpdateRule(neighborhood_size)

def argmin(l):
    if len(l) == 1:
        return 0
    best = l[0]
    best_i = 0
    for i in range(1, len(l)):
        if l[i] <= best:
            best_i = i
            best = l[i]
    return best_i

def evaluate_on_intervals(rule, intervals):
    error = 0.0
    for interval in intervals:
        error += evaluate_rule(rule, interval)
    return error

class GeneticTrainer:
    def __init__(self, args):
        self.neighborhood_size = args.neighbor
        self.mutate_amount = args.mutate
        self.n_generations = args.generations

    def train(self, intervals, graph):
        """Training a GA uses this algorithm:
        1. Generate an initial population of random-valued classifiers
        2. Repeat until convergence:
            a. Evaluate the fitness of all models
            b. Select the best N_keep models
            c. Use crossover to generate (N_pop - N_keep) new models
            d. Apply a chance to mutate to all models"""
        # Initialize search at cells with random parameters
        population = [make_random_rule(self.neighborhood_size) for _ in range(NUM_POPULATION)]

        # Iterate through generations:
        history = []
        for k in range(self.n_generations):
            print '== begin generation %s ==' % k
            # Evaluate the fitness of every CA in the population
            evaluations = []
            for update_rule in population:
                try:
                    evaluations.append(evaluate_on_intervals(update_rule, intervals))
                except OverflowError:
                    # If the CA is broken, assign it a really big error
                    evaluations.append(HUGE_NUMBER)

            # Track the best and average error over this generation
            error_mean = float(sum(evaluations)) / len(evaluations)
            error_min = min(evaluations)
            print 'errors: best %s, error_mean %s' % (error_min, error_mean)
            history.append((error_min, error_mean))

            # Select the best N_keep models for the new generation
            new_population = get_min_k(NUM_KEPT, population, evaluations)

            # Crossover the survivors to make the new generation
            random.shuffle(new_population)
            for i in range(0, len(new_population), 2):
                parent_a = new_population[i]
                parent_b = new_population[i + 1]
                new_population.append(parent_a.crossover(parent_b))

            # Create next generation by mutating the best performers.
            for model in population:
                model.mutate(MUTATION_RATE, MUTATION_DISTANCE)

            population = new_population

        plot_timeseries(history)
        best_index = argmin(evaluations)
        return population[best_index]

def parse_args():
    parser = make_ga_argument_parser()
    return parser.parse_args()

if __name__ == '__main__':
    # Create a genetic trainer
    args = parse_args()
    trainer = GeneticTrainer(args)
    data = Data.create_from_args(args)

    if args.num_folds is None:
        # Train it with a train-test split.
        basic_train(trainer, data)
    else:
        # Train it with CV trainer if -f flag is passed
        cross_validation_train(trainer, data)
