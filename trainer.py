from ca import Data

class Trainer:
    """Abstract trainer class."""
    def train(self, intervals, graph):
        pass

def basic_train(trainer, input_file, neighbor_file, split):
    data = Data(input_file, neighbor_file, split, split=split)
    trainer.train([data.partitions[0]], data.graph)

def cross_validation_train():
    # TODO - Leland
    pass