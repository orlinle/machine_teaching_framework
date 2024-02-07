from AbstractComponents import AbstractLearnerCentricTeachingLogic, AbstractVirtualLearner
from Virtual_learners.ComputationalModels import LinearLearner
import numpy as np
import random
from typing import List


class RandomDisagreeTeachingLogic(AbstractLearnerCentricTeachingLogic):

    def __init__(self, image_candidates_list: np.ndarray,
                 image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray,
                 virtual_learner: AbstractVirtualLearner):
        super().__init__(image_candidates_list=image_candidates_list, image_candidates_labels=image_candidates_labels,
                         image_candidate_paths=image_candidate_paths, virtual_learner=virtual_learner)

    def select_teaching_samples(self, batch_size: int) -> List[int]:
        """
        Simple random teaching logic returns a random image
        :return: index of chosen image
        """
        chosen_samples = []

        # chose images that vl label isn't aligned with ground truth (chooses random batch_size mistakes)
        for i, (sample, ground_truth_label) in enumerate(zip(self.image_candidates_list, self.image_candidates_labels)):
            if self.virtual_learner.predict(sample) != ground_truth_label:
                chosen_samples.append(i)
            if len(chosen_samples) == batch_size:
                break
        # if there are not enough mistaken images, fill the remaining teaching batch with random images
        if len(chosen_samples) < batch_size:
            num_images_to_fill = batch_size - len(chosen_samples)
            images_to_fill = [random.randint(0, len(self.image_candidates_labels) - 1) for _ in
                              range(num_images_to_fill)]
            chosen_samples.extend(images_to_fill)
        return chosen_samples

    def update_virtual_learner(self, samples: np.ndarray, human_labels: np.ndarray):
        self.virtual_learner.fit(samples=samples, human_labels=human_labels)


# class ConfidenceDisagreementTeachingLogic(AbstractLearnerCentricTeachingLogic):
#     def __init__(self, image_candidates_list: np.ndarray,
#                  image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray, virtual_learner: LinearLearner):
#         super().__init__(image_candidates_list=image_candidates_list, image_candidates_labels=image_candidates_labels, image_candidate_paths=image_candidate_paths, virtual_learner=virtual_learner)
#
#     def select_teaching_samples(self, batch_size: int) -> List[int]:
#         # TODO implement!!
#         """
#         Simple random teaching logic returns a random image
#         :return: index of chosen image
#         """
#         pass

class UncertaintyBasedTeachingLogic(AbstractLearnerCentricTeachingLogic):
    def __init__(self, image_candidates_list: np.ndarray,
                 image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray,
                 virtual_learner: LinearLearner):
        super().__init__(image_candidates_list=image_candidates_list, image_candidates_labels=image_candidates_labels,
                         image_candidate_paths=image_candidate_paths, virtual_learner=virtual_learner)

    def select_teaching_samples(self, batch_size: int) -> List[int]:
        # TODO implement!!
        """
        Simple random teaching logic returns a random image
        :return: index of chosen image
        """
        pass
