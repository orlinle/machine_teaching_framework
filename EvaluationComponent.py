from AbstractComponents import AbstractEvaluationComponent, AbstractTeacher
import numpy as np
import matplotlib.pyplot as plt


class BatchTeachingEvaluator(AbstractEvaluationComponent):
    def __init__(self, teacher: AbstractTeacher, eval_results_path: str, before_test_set_size: int,
                 after_test_set_size: int):
        super().__init__(teacher=teacher, eval_results_path=eval_results_path)
        self.before_test_set_size = before_test_set_size
        self.after_test_set_size = after_test_set_size
        self.before_test_set = super().save_images_aside_for_test_set(before_test_set_size)
        self.after_test_set = super().save_images_aside_for_test_set(after_test_set_size)

    def _get_session_evaluation(self, session_dataset):
        correct_answers_count = 0
        before_eval_images = session_dataset['image_paths']
        before_eval_labels = session_dataset['image_labels']
        for image, ground_truth_label in zip(before_eval_images, before_eval_labels):
            human_label = self.teacher.query_human_learner(image)
            if human_label == ground_truth_label:
                correct_answers_count += 1
        return correct_answers_count

    def _evaluate_training_accuracy(self, chosen_indices, human_labels):
        chosen_images_ground_truth_labels = super().get_values_at_indices(indices=chosen_indices,
                                                                          values=self.teacher.teaching_logic.image_candidates_labels)
        correct_human_predictions = sum(
            1 for elem1, elem2 in zip(chosen_images_ground_truth_labels, human_labels) if elem1 == elem2)
        return np.round(correct_human_predictions / len(human_labels), 2)

    def _run_teaching_session(self):
        print("Let's start with an evaluation session to test your performance before the teaching begins")
        pre_teaching_accuracy = np.round(self._get_session_evaluation(self.before_test_set) / self.before_test_set_size, 2)
        print("Great! We finished with the evaluation")
        print(
            "Now we move on to a teaching session where you will be shown images and will need to predict their labels to the best of your ability.")
        print(
            "If you don't know the answer - that's ok! Just do your best. Once you answer, we will tell you if your answer was correct. Try to remember for next time..")
        chosen_indices, human_labels = self.teacher.perform_teaching_process()
        training_accuracy = self._evaluate_training_accuracy(chosen_indices=chosen_indices, human_labels=human_labels)
        print(
            "Now we'll see how much you learned. You will be given a few more images to label and we will see how many you label correctly. Notice - for this stage we will not tell you after each image if you were correct or not.")
        post_teaching_accuracy = np.round(self._get_session_evaluation(self.after_test_set) / self.after_test_set_size, 2)
        training_samples_count = len(human_labels)
        return pre_teaching_accuracy, training_accuracy, post_teaching_accuracy, training_samples_count

    def evaluate_teaching_framework(self):
        pre_teaching_accuracy, training_accuracy, post_teaching_accuracy, training_samples_count = self._run_teaching_session()
        eval_names = ['pre_teaching_accuracy', "training_accuracy", "post_teaching_accuracy"]
        accuracy_rates = [pre_teaching_accuracy, training_accuracy, post_teaching_accuracy]
        sample_sizes = [self.before_test_set_size, training_samples_count, self.after_test_set_size]
        plt.bar(eval_names, accuracy_rates)
        for i, (eval_name, accuracy) in enumerate(zip(eval_names, accuracy_rates)):
            plt.text(i, accuracy / 2, f'n={sample_sizes[i]}', ha='center', va='center', color='white')

        plt.title("Accuracy before, during and after batch teaching process")
        plt.savefig(self.eval_results_path)
        plt.show()
        plt.close()
