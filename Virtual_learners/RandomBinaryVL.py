from AbstractComponents import AbstractVirtualLearner
import random


class RandomBinaryVirtualLearner(AbstractVirtualLearner):
    super().__init__()
    def predict(self, sample) -> int:
        return random.choice([0, 1])
