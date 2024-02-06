from AbstractComponents import AbstractTeachingLogic, AbstractTeacher
from random import choice

class RandomTeachingLogic(AbstractTeachingLogic):
    def __init__(self, teaching_set_pool_size: int):
        super().__init__()
        self.teaching_set_pool_size = teaching_set_pool_size

    def select_teaching_sample(self) -> int:
        """
        Simple random teaching logic returns a random image
        :return: index of chosen image
        """
        return choice(range(self.teaching_set_pool_size))
