from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict
import numpy as np
from PIL import Image
import os, random
import matplotlib.pyplot as plt


class AbstractImageProcessor(ABC):
    """
    Abstract base class for performing feature extraction on imges.

    Attributes:
        raw_data_path (str): Path to the raw data folder.
        extracted_features_path (str): Path to the folder where extracted features will be saved.
        teaching_images: List of teaching images.
        teaching_images_features: Features extracted from teaching images.
        teaching_images_labels: Labels of teaching images.
    """
    def __init__(self, raw_data_path: str, extracted_features_path: str, teaching_images, teaching_images_features,
                 teaching_images_labels):
        """
        Initialize the AbstractImageProcessor.

        Args:
            raw_data_path (str): Path to the raw data folder.
            extracted_features_path (str): Path to the folder where extracted features will be saved.
            teaching_images: List of teaching images.
            teaching_images_features: Features extracted from teaching images.
            teaching_images_labels: Labels of teaching images.
        """
        self.raw_data_path = raw_data_path
        self.extracted_features_path = extracted_features_path
        self.teaching_images = teaching_images
        self.teaching_images_features = teaching_images_features
        self.teaching_images_labels = teaching_images_labels

    @abstractmethod
    def perform_feature_extraction(self):
        """
        Abstract method to be implemented by subclasses.
        Given a path to a folder with images, create a folder of image representations.
        """
        pass


class AbstractVirtualLearner(ABC):
    """
    Abstract base class for virtual learner models.

    Attributes:
        None
    """
    def __init__(self):
        """
        Initialize the AbstractVirtualLearner.
        """
        pass

    @abstractmethod
    def predict(self, sample) -> Union[str, int]:
        """
        Abstract method to be implemented by subclasses.
        Given a sample image, return the approximation of human Learner prediction.

        Args:
            sample: The sample image to predict.

        Returns:
            Union[str, int]: The predicted label.
        """
        pass

    @abstractmethod
    def fit(self, samples: np.ndarray, human_labels: np.ndarray):
        """
        Update the virtual learner according to responses of human Learner to new samples.

        Args:
            samples (np.ndarray): Samples shown to human Learner.
            human_labels (np.ndarray): Labels given by human.
        """
        pass


class AbstractTeachingLogic(ABC):
    """
    Abstract base class for teaching logic models.

    Attributes:
        image_candidates_list (np.ndarray): Array of image candidates.
        image_candidates_labels (np.ndarray): Array of labels for image candidates.
        image_candidate_paths (np.ndarray): Array of paths for image candidates.
    """
    def __init__(self, image_candidates_list: np.ndarray,
                 image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray):
        """
        Initialize the AbstractTeachingLogic.

        Args:
            image_candidates_list (np.ndarray): Array of image candidates.
            image_candidates_labels (np.ndarray): Array of labels for image candidates.
            image_candidate_paths (np.ndarray): Array of paths for image candidates.
        """
        self.image_candidates_list = image_candidates_list
        self.image_candidates_labels = image_candidates_labels
        self.image_candidate_paths = image_candidate_paths


    def reset_all_attributes(self,image_candidates_list: np.ndarray,
                 image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray):
        """
        Reset all attributes of the teaching logic.

        Args:
            image_candidates_list (np.ndarray): Array of image candidates.
            image_candidates_labels (np.ndarray): Array of labels for image candidates.
            image_candidate_paths (np.ndarray): Array of paths for image candidates.
        """
        self.image_candidates_list = image_candidates_list
        self.image_candidates_labels = image_candidates_labels
        self.image_candidate_paths = image_candidate_paths

    @abstractmethod
    def select_teaching_samples(self, batch_size: int) -> List[int]:
        """
        Abstract method to be implemented by subclasses.
        Choose the best teaching samples to show the learner.

        Args:
            batch_size (int): The number of samples to select.

        Returns:
            List[int]: An array of indices of the chosen images.
        """
        pass


class AbstractLearnerCentricTeachingLogic(AbstractTeachingLogic):
    """
    Abstract base class for learner-centric teaching logic models.

    Attributes:
        virtual_learner (AbstractVirtualLearner): Instance of a virtual learner.
    """
    def __init__(self, image_candidates_list: np.ndarray,
                 image_candidates_labels: np.ndarray, image_candidate_paths: np.ndarray,
                 virtual_learner: AbstractVirtualLearner):
        """
        Initialize the AbstractLearnerCentricTeachingLogic.

        Args:
            image_candidates_list (np.ndarray): Array of image candidates.
            image_candidates_labels (np.ndarray): Array of labels for image candidates.
            image_candidate_paths (np.ndarray): Array of paths for image candidates.
            virtual_learner (AbstractVirtualLearner): Instance of a virtual learner.
        """
        super().__init__(image_candidates_list=image_candidates_list, image_candidates_labels=image_candidates_labels,
                         image_candidate_paths=image_candidate_paths)
        self.virtual_learner = virtual_learner

    @abstractmethod
    def update_virtual_learner(self, samples: np.ndarray, human_labels: np.ndarray):
        """
        Abstract method to be implemented by subclasses.
        Update the virtual learner based on new samples and human labels.

        Args:
            samples (np.ndarray): Samples shown to the human learner.
            human_labels (np.ndarray): Labels provided by the human learner.
        """
        self.virtual_learner.fit(samples=samples, human_labels=human_labels)


class AbstractTeacher(ABC):
    """
    Abstract base class for teacher models.

    Attributes:
        teaching_logic (AbstractTeachingLogic): Instance of teaching logic.
        human_query_text (str): Text for querying the human learner.
        total_teaching_samples (int): Total number of teaching samples.
    """
    def __init__(self, teaching_logic: AbstractTeachingLogic, human_query_text: str, total_teaching_samples: int):
        """
        Initialize the AbstractTeacher.

        Args:
            teaching_logic (AbstractTeachingLogic): Instance of teaching logic.
            human_query_text (str): Text for querying the human learner.
            total_teaching_samples (int): Total number of teaching samples.
        """

        self.teaching_logic = teaching_logic
        self.human_query_text = human_query_text
        self.total_teaching_samples = total_teaching_samples

    @abstractmethod
    def perform_teaching_process(self) -> Tuple[List[int], List[str]]:
        """
        Abstract method to be implemented by subclasses.
        Perform the teaching process and return the chosen image indices and human learner predictions.

        Returns:
            Tuple[List[int], List[str]]: Tuple containing a list of integers representing the indices of the chosen
            images throughout the teaching process and a list of strings representing the predictions given from the
            human learner throughout the teaching process.
        """
        pass

    def query_human_learner(self, chosen_image_path: str) -> str:
        """
        Query the human learner with a chosen image.

        Args:
            chosen_image_path (str): Path of the chosen image.

        Returns:
            str: Label provided by the human learner.
        """
        image = Image.open(chosen_image_path)
        print(self.human_query_text)
        plt.imshow(image)
        plt.axis('off')  # Turn off axis
        plt.show()
        human_learner_label = input()
        return human_learner_label


class AbstractEvaluationComponent(ABC):
    """
    Abstract base class for evaluation components.

    Attributes:
        teacher (AbstractTeacher): Instance of the teacher model.
        eval_results_path (str): Path to save evaluation results.
    """
    def __init__(self, teacher: AbstractTeacher, eval_results_path: str):
        """
        Initialize the AbstractEvaluationComponent.

        Args:
            teacher (AbstractTeacher): Instance of the teacher model.
            eval_results_path (str): Path to save evaluation results.
        """
        self.teacher = teacher
        self.eval_results_path = eval_results_path

    @abstractmethod
    def evaluate_teaching_framework(self):
        """
        Abstract method to be implemented by subclasses.
        Evaluate the teaching framework.
        """
        pass

    @staticmethod
    def get_values_at_indices(indices, values):
        """
        Get values at given indices from a list.

        Args:
            indices (List[int]): List of indices.
            values (List): List of values.

        Returns:
            np.ndarray: Array of values at given indices.
        """
        return np.array([values[i] for i in indices])

    @staticmethod
    def remove_given_indices(values, indices_to_remove):
        """
        Remove values at given indices from a list.

        Args:
            values (List): List of values.
            indices_to_remove (List[int]): List of indices to remove.

        Returns:
            np.ndarray: Array of values after removing indices.
        """
        return np.array([value for i, value in enumerate(values) if i not in indices_to_remove])

    def _modify_original_dataset_remove_test_images(self, saved_images_indices):
        """
        Modify the original dataset by removing images used for the test set.

        Args:
            saved_images_indices (List[int]): List of indices of images used for the test set.
        """
        modified_image_candidates_list, modified_image_candidates_labels, modified_image_candidate_paths = [
            self.remove_given_indices(orig_dataset, saved_images_indices) for orig_dataset in
            [self.teacher.teaching_logic.image_candidates_list, self.teacher.teaching_logic.image_candidates_labels,
             self.teacher.teaching_logic.image_candidate_paths]]
        self.teacher.teaching_logic.reset_all_attributes(image_candidate_paths=modified_image_candidate_paths,
                                                         image_candidates_list=modified_image_candidates_list,
                                                         image_candidates_labels=modified_image_candidates_labels)

    def save_images_aside_for_test_set(self, num_images: int) -> Dict[str, np.ndarray]:
        """
        Save images aside for the test set and modify the original dataset accordingly.

        Args:
            num_images (int): Number of images to save for the test set.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing image paths, features, and labels of the test set.
        """
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
