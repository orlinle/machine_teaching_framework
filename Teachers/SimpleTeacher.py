from AbstractComponents import AbstractTeacher, AbstractImageProcessor, AbstractTeachingLogic
from Data_processors.ResNet50FeatureExtractor import ResNet50FeatureExtractor
from Teaching_logics.RandomTeachingLogic import RandomTeachingLogic
from PIL import Image
import matplotlib.pyplot as plt

class SimpleTeacher(AbstractTeacher):
    def __init__(self, data_processor: AbstractImageProcessor, teaching_logic: AbstractTeachingLogic, max_teaching_rounds: int, learner_query_text: str):
        self.data_processor = data_processor
        self.teaching_logic = teaching_logic
        super().__init__(data_processor=self.data_processor, teaching_logic=self.teaching_logic)
        self.max_teaching_rounds = max_teaching_rounds
        self.learner_query_text = learner_query_text
        self.teaching_iterations_counter = 0
        self.chosen_images = []

    def teaching_termination_logic(self) -> bool:
        return True if self.teaching_iterations_counter == self.max_teaching_rounds else False

    def _show_image_to_learner(self, chosen_image_idx):
        chosen_image = self.data_processor.teaching_images[chosen_image_idx]
        image = Image.open(chosen_image)
        print(self.learner_query_text)
        plt.imshow(image)
        plt.axis('off')  # Turn off axis
        plt.show()

    def perform_teaching_transaction(self):
        correct_human_responses = 0
        while (self.teaching_termination_logic()):
            chosen_image_idx = self.teaching_logic.select_teaching_sample()
            self._show_image_to_learner(chosen_image_idx=chosen_image_idx)
            chosen_image_ground_truth = self.data_processor.teaching_images_labels[chosen_image_idx]
            human_learner_label = input()
            if int(human_learner_label) == chosen_image_ground_truth:
                print("You are correct!")
                correct_human_responses += 1
            else:
                print(f"Mistake. True label is {chosen_image_ground_truth}")
        print(f"Teaching session is over. You were correct {correct_human_responses / self.max_teaching_rounds * 100}% percent of the time. Good job!")

