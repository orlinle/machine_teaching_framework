from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict
from configs import Teacher_configs
import numpy as np
from PIL import Image
import os, random
import matplotlib.pyplot as plt


class AbstractImageProcessor(ABC):
    def __init__(self, raw_data_path: str, extracted_features_path: str, teaching_images, teaching_images_features,
                 teaching_images_labels):
        # Common initialization for teaching logic
        self.raw_data_path = raw_data_path
        self.extracted_features_path = extracted_features_path
        self.teaching_images = teaching_images
        self.teaching_images_features = teaching_images_features
        self.teaching_images_labels = teaching_images_labels

    @abstractmethod
    def perform_feature_extraction(self):
        """
        Abstract method to be implemented by subclasses.
        Given a path to a folder with images, create a folder of image representations with corresponding names
        """
        pass


class AbstractVirtualLearner(ABC):
    def __init__(self):
        # Common initialization for teaching logic
        pass

    @abstractmethod
    def predict(self, sample) -> Union[str, int]:
        """
        Abstract method to be implemented by subclasses.
        Given a sample image, return the approximation of human Learner prediction.
        """
        pass

    @abstractmethod
    def fit(self, samples: np.ndarray, human_labels: np.ndarray):
        """
        This function allows the Teacher to update the VL according to responses of human Learner to new samples
        :param samples: samples shown to human Learner
        :param human_labels: Labels given by human
        """
        pass


class AbstractTeachingLogic(ABC):
    def __init__(self, image_candidates_list: np.ndarray,
                 image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray):
        self.image_candidates_list = image_candidates_list
        self.image_candidates_labels = image_candidates_labels
        self.image_candidate_paths = image_candidate_paths


    def reset_all_attributes(self,image_candidates_list: np.ndarray,
                 image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray):
        self.image_candidates_list = image_candidates_list
        self.image_candidates_labels = image_candidates_labels
        self.image_candidate_paths = image_candidate_paths

    @abstractmethod
    def select_teaching_samples(self, batch_size: int) -> List[int]:
        """
        Abstract method to be implemented by subclasses.
        Choose the best teaching samples to show the learner.
        :return: an array of indices of the chosen images
        """
        pass


class AbstractLearnerCentricTeachingLogic(AbstractTeachingLogic):
    def __init__(self, image_candidates_list: np.ndarray,
                 image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray,
                 virtual_learner: AbstractVirtualLearner):
        super().__init__(image_candidates_list=image_candidates_list, image_candidates_labels=image_candidates_labels,
                         image_candidate_paths=image_candidate_paths)
        self.virtual_learner = virtual_learner

    @abstractmethod
    def update_virtual_learner(self, samples: np.ndarray, human_labels: np.ndarray):
        """
        Abstract method to be implemented by subclasses.
        Choose the best teaching samples to show the learner.
        :return: an array of indices of the chosen images
        """
        self.virtual_learner.fit(samples=samples, human_labels=human_labels)


class AbstractTeacher(ABC):
    def __init__(self, teaching_logic: AbstractTeachingLogic, human_query_text: str, total_teaching_samples: int):
        self.teaching_logic = teaching_logic
        self.human_query_text = human_query_text
        self.total_teaching_samples = total_teaching_samples
        self.config = Teacher_configs

    # @abstractmethod
    # def _initialize_attributes(self):
    #     """
    #     A method to ensure each Teacher initialized it's attributes througth the Teacher config file
    #     """
    #     pass

    @abstractmethod
    def perform_teaching_process(self) -> Tuple[List[int], List[str]]:
        """
        This method encapsulates the entire teaching process. :return: List[int]: a list of integers representing the
        indices of the images chosen throughout the teaching process List[str]: a list of strings representing the
        predictions given from the human learner throughout the teaching process
        """
        pass

    def query_human_learner(self, chosen_image_path: str) -> str:
        image = Image.open(chosen_image_path)
        print(self.human_query_text)
        plt.imshow(image)
        plt.axis('off')  # Turn off axis
        plt.show()
        human_learner_label = input()
        return human_learner_label


class AbstractEvaluationComponent(ABC):
    def __init__(self, teacher: AbstractTeacher, eval_results_path: str):
        self.teacher = teacher
        self.eval_results_path = eval_results_path

    @abstractmethod
    def evaluate_teaching_framework(self):
        pass

    @staticmethod
    def get_values_at_indices(indices, values):
        return np.array([values[i] for i in indices])

    @staticmethod
    def _remove_given_indices(values, indices_to_remove):
        return np.array([value for i, value in enumerate(values) if i not in indices_to_remove])

    def _modify_original_dataset_remove_test_images(self, saved_images_indices):
        modified_image_candidates_list, modified_image_candidates_labels, modified_image_candidate_paths = [
            self._remove_given_indices(orig_dataset, saved_images_indices) for orig_dataset in
            [self.teacher.teaching_logic.image_candidates_list, self.teacher.teaching_logic.image_candidates_labels,
             self.teacher.teaching_logic.image_candidate_paths]]
        self.teacher.teaching_logic.reset_all_attributes(image_candidate_paths=modified_image_candidate_paths,
                                                         image_candidates_list=modified_image_candidates_list,
                                                         image_candidates_labels=modified_image_candidates_labels)

    def save_images_aside_for_test_set(self, num_images: int) -> Dict[str, np.ndarray]:
        total_images_count = len(self.teacher.teaching_logic.image_candidates_labels)
        saved_images_indices = [random.randint(0, total_images_count - 1) for _ in range(num_images)]
        test_set = {
            'image_paths': self.get_values_at_indices(saved_images_indices,
                                                       self.teacher.teaching_logic.image_candidate_paths),
            'image_features': self.get_values_at_indices(saved_images_indices,
                                                          self.teacher.teaching_logic.image_candidates_list),
            'image_labels': self.get_values_at_indices(saved_images_indices,
                                                        self.teacher.teaching_logic.image_candidates_labels)
        }
        self._modify_original_dataset_remove_test_images(saved_images_indices)
        return test_set
