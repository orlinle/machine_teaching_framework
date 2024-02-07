from AbstractComponents import AbstractTeachingLogic
# from modAL.density import information_density
import random, numpy as np
from typing import List, Union


class RandomTeachingLogic(AbstractTeachingLogic):
    def __init__(self, image_candidates_list: np.ndarray,
                 image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray):
        super().__init__(image_candidates_list=image_candidates_list, image_candidates_labels=image_candidates_labels, image_candidate_paths=image_candidate_paths)

    def select_teaching_samples(self, batch_size: int) -> List[int]:
        """
        Simple random teaching logic returns a random image
        :return: index of chosen image
        """
        return [random.randint(0, len(self.image_candidates_labels) - 1) for _ in range(batch_size)]


# class InformationBasedTeachingLogic(AbstractTeachingLogic):
#
#     def __init__(self, image_candidates_list: np.ndarray,
#                  image_candidates_labels: np.ndarray):
#         super().__init__(image_candidates_list=image_candidates_list, image_candidates_labels=image_candidates_labels, image_candidate_paths=image_candidate_paths)
#
#     def select_teaching_samples(self, batch_size: int) -> List[int]:
#         euclidean_density = information_density(self.image_candidates_list, 'euclidean')
#         # TODO implement!!
#         pass
