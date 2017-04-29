from ca import Data, plot_error

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

def cross_validation_train(trainer, data):
    """Train k models on k different splits of data.partitions.

    Arguments
    trainer -- instance of Trainer
    data -- instance of Data. Assume that data.partitions is a list of k training folds
    """
    #
    # TODO - Leland
    print 'cross_validation_train'
