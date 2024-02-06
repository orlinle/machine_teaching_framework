from AbstractComponents import AbstractTeachingLogic, AbstractVirtualLearner
from typing import List
from random import choice


class SimpleLearnerCentricTeachingLogic(AbstractTeachingLogic):
    """
    Simple example of teaching logic utilizing a Virtual Learner.
    This logic Chooses a sample according to Virtual Learner's agreement with ground truth labels.
    """

    def __init__(self, virtual_learner: AbstractVirtualLearner, image_candidates_list: List[List[int]], image_candidates_labels: List[int]):
        """
        :param virtual_learner: Virtual Learner that is handled and updated by the Teacher. This teaching logic does
        not depend on the specific implementation of the virtual Learner
        :param image_candidates_list: the list of candidate images to choose from
        :param image_candidates_labels: the image labels
        """
        super().__init__()
        self.virtual_learner = virtual_learner
        self.image_candidates_list = image_candidates_list
        self.image_candidates_labels = image_candidates_labels

    def select_teaching_sample(self) -> int:
        """
        The teaching logic examines each candidate sample and checks whether the Virtual Learner agrees
        with the ground truth label. If not - the sample is chosen as a teaching sample in order to teach it to the
        human Learner. If virtual learner correctly predicts all samples this is an edge case. In this sample we chose
        to handle it by choosing a random sample.
        :param image_candidates_list: images are assumed to be vectors of ints, therefore this will be a list of vectors
        :param image_candidates_labels: each image label is represented by an int, therefore this will be a list of ints
        :return: chosen image index
        """
        for sample, ground_truth_label in zip(self.image_candidates_list, self.image_candidates_labels):
            if self.virtual_learner.predict(sample) != ground_truth_label:
                return sample
        return choice(range(len(self.image_candidates_list)))
