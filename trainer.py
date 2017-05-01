from ca import Data, plot_error, pearson_correlation

class Trainer:
    """Abstract trainer class."""
    def train(self, intervals, graph):
        """Abstract train method.
        Returns value: an instance of UpdateRule"""
        pass

def basic_train(trainer, data):
    # Assume that data comes in a train / test split. Train on the first partition.
    best_rule = trainer.train([data.partitions[0]], data.graph)

    # [TODO] evaluate on the second

    plot_error(best_rule, data.partitions[0], 0)
    plot_error(best_rule, data.partitions[0], 1)
    plot_error(best_rule, data.partitions[0], 2)

"""Train k models on k different splits of data.partitions.
Arguments
trainer -- instance of Trainer
data -- instance of Data. Assume that data.partitions is a list of k training folds
"""
def cross_validation_train(trainer, data):
    
    print('cross_validation_train')
    
    num_partitions = len(data.partitions)
    rules = []
    for i in range(num_partitions):
        training_partitions = []
        for n in range(num_partitions):
            if n != i:
                training_partitions.append(data.partitions[i])
        current_rule = trainer.train(training_partitions, data.graph)
        rules.append(current_rule)
    # Calculate average pearson correlation for each fold
    correlations = []
    for i in range(len(rules)):
        current_rule = rules[i]
        testing_partition = data.partitions[i]
        correlations.append(pearson_correlation(current_rule, testing_partition))
    # Find rule with best pearson correlation 
    best_index = 0
    best_correlation = -10
    for i in range(len(rules)):
        if correlations[i] > best_correlation:
            best_correlation = correlations[i]
            best_index = i
    # Applying rule with best pearson correlation, plot output for one city  
    print('best correlation:')
    print(best_correlation)     
    best_rule = rules[best_index]
    testing_partition = data.partitions[best_index]
    plot_error(best_rule, testing_partition, 0)
  
    