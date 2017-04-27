from ca import make_argument_parser as make_ca_argument_parser, Data, UpdateRule, CellularAutomaton, evaluate_rule
from trainer import Trainer, basic_train
import random
import numpy as np
import matplotlib.pyplot as plt

POPULATION_SIZE = 100
SURVIVOR_RATIO = 0.2 # on average 20% of the population survives
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

# Comment or uncommont to change the pruning strategy
USE_SELECTION_STRATEGY_A = True
# USE_SELECTION_STRATEGY_A = False

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
        # Initialize search at cells with random parameters
        population = [make_random_rule(self.neighborhood_size) for _ in range(POPULATION_SIZE)]

        # Iterate through generations:
        history = []
        for k in range(self.n_generations):
            print '== begin generation %s ==' % k
            # Evaluate every CA in the population
            evaluations = []
            for update_rule in population:
                try:
                    evaluations.append(evaluate_on_intervals(update_rule, intervals))
                except OverflowError:
                    # Just append a really big weight if the CA is broken
                    evaluations.append(HUGE_NUMBER)

            # Calculate the best and average error over this generation

            average = float(sum(evaluations)) / len(evaluations)
            best_index = argmin(evaluations)
            if USE_SELECTION_STRATEGY_A:
                new_population = [population[best_index]]
            print 'errors: best %s, average %s' % (evaluations[best_index], average)
            history.append((evaluations[best_index], average))

            # Prune population according to probability of survival
            # TODO is inverse error a good fitness?
            if not USE_SELECTION_STRATEGY_A:
                fitnesses = [1/float(error) if error != 0 else 10000 for error in evaluations]
                sum_fitness = sum(fitnesses)
                new_population = []
                for update_rule, fitness in zip(population, fitnesses):
                    # Probability of success is fraction of fitness of total
                    if float(fitness) / sum_fitness > random.random() * SURVIVOR_RATIO:
                        new_population.append(update_rule)

            if len(new_population) == 0:
                print 'ah geez, no one lived ):'
                new_population.append(population[0])

            # Create next generation by mutating the best performers.
            generation_size = len(new_population)
            for i in range(POPULATION_SIZE - generation_size):
                # Cycle through survivors to be parents
                parent = new_population[i % generation_size]
                new_population.append(parent.mutate(self.mutate_amount)) # TODO consider making this smaller throughout training

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

    # Train it with a train-test split.
    # TODO add support for cross validation
    basic_train(trainer, args.input_file, args.neighbor_file, args.split)
