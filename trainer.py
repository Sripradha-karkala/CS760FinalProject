from ca import Data

class Trainer:
    """Abstract trainer class."""
    def train(self, intervals):
        pass

def basic_train(input_file, split, trainer):
    data = Data(input_file, split)
    trainer.train([data.train_data])
