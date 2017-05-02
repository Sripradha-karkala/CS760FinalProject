from ca import make_argument_parser as make_ca_argument_parser, Data, UpdateRule, CellularAutomaton, evaluate_rule, DataType
from trainer import Trainer, basic_train, cross_validation_train
import random
import numpy as np
import math
import matplotlib.pyplot as plt

# We need to catch overflow errors and handle them, instead of just printing.
np.seterr(all='raise')

NUM_POPULATION = 100
NUM_KEPT = 40 # keep this many for the next generation. must be even.
MUTATION_RATE = 0.20 # the proportion of chromosomes that are mutated

# USE_COSINE_SIMILARITY = True
USE_COSINE_SIMILARITY = False

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
            help = 'Specify the mutation rate, the proportion of genes which are mutated each generation.')
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

def predict_values(rule, interval):
    """Creates a numpy array values, where values[i][j] represents the predicted value of city j at time i"""
    ca = CellularAutomaton(interval[0], rule)
    values = np.zeros((len(interval), len(interval[0])))
    values[0] = interval[0]
    for t in range(1, len(interval)):
        ca.update()
        values[t] = ca.get_values()[:, 0]
    return values

def cosine_similarity(A, B):
    # Numerator term
    numerator = 0.0
    for a, b in zip(A, B):
        numerator += a * b

    # Denominator term A
    denom_a = 0.0
    for a in A:
        denom_a += a ** 2
    denom_a = math.sqrt(denom_a)

    # Denominator term B
    denom_b = 0.0
    for b in B:
        denom_b += b ** 2
    denom_b = math.sqrt(denom_b)

    return numerator / (denom_a * denom_b)

def evalaute_cosine_similarity(rule, interval):
    """rule -- Instance of UpdateRule
    interval -- numpy matrix where interval[i][j] represents the value of city j at time i"""
    predicted = predict_values(rule, interval)

    # Sum cosine similarities for every city
    similarity = 0.0
    for i in range(len(interval[0])):
        city_values = interval[:, i]
        predicted_values = predicted[:, i]
        similarity += cosine_similarity(city_values, predicted_values)
    return similarity

def evaluate_on_intervals(rule, intervals):
    error = 0.0
    for interval in intervals:
        if USE_COSINE_SIMILARITY:
            error += 1 - evalaute_cosine_similarity(rule, interval)
        else:
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
                except FloatingPointError:
                    # If the CA is broken, assign it a really big error
                    evaluations.append(float('inf'))
                # except KeyboardInterrupt:
                #     print 'foofoo'

            # Track the best and average error over this generation
            error_mean = float(sum(evaluations)) / len(evaluations)
            error_min = min(evaluations)
            print 'errors: best %s, error_mean %s' % (error_min, error_mean)
            history.append((error_min, error_mean))

            # Select the best N_keep models for the new generation
            survivors = get_min_k(NUM_KEPT, population, evaluations)

            # Crossover the survivors to make the new generation
            random.shuffle(survivors)
            children = []
            for j in range(NUM_POPULATION - NUM_KEPT):
                i = (j * 2) % NUM_KEPT
                parent_a = survivors[i]
                parent_b = survivors[i + 1]
                children.append(parent_a.crossover(parent_b))

            # Create next generation by mutating the best performers.
            # Elitism: Only mutate the children
            for model in children:
                model.mutate(MUTATION_RATE)

            population = survivors + children

        # plot_timeseries(history)
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
