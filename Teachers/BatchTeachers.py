from AbstractComponents import AbstractTeacher, AbstractTeachingLogic
import numpy as np


class SimpleBatchTeacher(AbstractTeacher):
    def __init__(self, teaching_logic: AbstractTeachingLogic, human_query_text: str, total_teaching_samples: int):
        super().__init__(teaching_logic=teaching_logic, human_query_text=human_query_text,
                         total_teaching_samples=total_teaching_samples)
        self.total_teaching_samples = total_teaching_samples

    def _get_values_at_indices(self, indices, values):
        return np.array([values[i] for i in indices])

    def perform_teaching_process(self):
        chosen_samples = self.teaching_logic.select_teaching_samples(self.total_teaching_samples)
        chosen_image_paths = self._get_values_at_indices(indices=chosen_samples,
                                                         values=self.teaching_logic.image_candidate_paths)
        chosen_image_true_label = self._get_values_at_indices(indices=chosen_samples,
                                                         values=self.teaching_logic.image_candidates_labels)
        human_labels = []
        for sample, true_label in zip(chosen_image_paths, chosen_image_true_label):
            human_label = super().query_human_learner(sample)
            human_labels.append(human_label)
            if human_label == true_label:
                print("You are correct!")
            else:
                print(f"Not quite. The correct answer is {true_label}")
        return chosen_samples, human_labels
