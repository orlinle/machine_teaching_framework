from abc import ABC, abstractmethod
from typing import Union


class AbstractImageProcessor(ABC):
    def __init__(self, raw_data_path: str, extracted_features_path: str, teaching_images, teaching_images_features, teaching_images_labels):
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

class AbstractTeachingLogic(ABC):
    def __init__(self):
        # Common initialization for teaching logic
        pass

    @abstractmethod
    def select_teaching_sample(self) -> int:
        """
        Abstract method to be implemented by subclasses.
        Given a list of image candidates, return the best sample to show the learner.
        """
        pass

class AbstractTeacher(ABC):
    def __init__(self, data_processor: AbstractImageProcessor, teaching_logic: AbstractTeachingLogic):
        self.data_processor = data_processor
        self.teaching_logic = teaching_logic

    @abstractmethod
    def teaching_termination_logic(self) -> bool:
        """
        A method holding the logic of when the teaching process should be terminated.
        :return: True if teaching should be terminated, otherwise False
        """
        pass

    @abstractmethod
    def perform_teaching_transaction(self):
        """
        This method encapsulates what the Teacher does in a single teaching transaction
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
