from ca import make_argument_parser, Data, UpdateRule, CellularAutomaton, evaluate_rule
import random
import numpy as np

POPULATION_SIZE = 100
N_GENERATIONS = 100
PERTURB_AMOUNT = 0.1 # I dunno
SURVIVOR_RATIO = 0.2 # on average 20% of the population survives

# Comment or uncommont to change the pruning strategy
USE_SELECTION_STRATEGY_A = True
# USE_SELECTION_STRATEGY_A = False

def make_random_rule(neighborhood_size):
    return UpdateRule(neighborhood_size)

def genetic_train(args):
    neighborhood_size = args.neighbor
    data = Data(args.input_file, args.split)

    # Initialize search at cells with random parameters
    population = [make_random_rule(neighborhood_size) for _ in range(POPULATION_SIZE)]

    # Iterate through generations:
    for k in range(N_GENERATIONS):
        print '== begin generation %s ==' % k
        # Evaluate every CA in the population
        evaluations = []
        for update_rule in population:
            try:
                evaluations.append(evaluate_rule(update_rule, data.train_data))
            except OverflowError:
                # Just append a really big weight if the CA is broken
                evaluations.append(10000)

        # Calculate the best and average error over this generation

        average = float(sum(evaluations)) / len(evaluations)
        best_index = 0
        for i in range(len(population)):
            if evaluations[i] <= evaluations[best_index]:
                best_index = i
        if USE_SELECTION_STRATEGY_A:
            new_population = [population[best_index]]
        print 'errors: best %s, average %s' % (evaluations[best_index], average)

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

        # Create next generation by perturbing the best performers.
        generation_size = len(new_population)
        for i in range(POPULATION_SIZE - generation_size):
            # Cycle through survivors to be parents
            parent = new_population[i % generation_size]
            new_population.append(parent.perturb(PERTURB_AMOUNT)) # TODO consider making this smaller throughout training

        population = new_population

def parse_args():
    parser = make_argument_parser()
    return parser.parse_args()

if __name__ == '__main__':
    genetic_train(parse_args())