from Data_processors.ResNet50FeatureExtractor import ResNet50FeatureExtractor
from Teachers.SimpleTeacher import SimpleTeacher
from Teaching_logics.RandomTeachingLogic import RandomTeachingLogic


def simple_teaching_process():
    raw_data_path = "Datasets/Binary_mnist"
    extracted_features_path = "Dataset_features/Binary_mnist"
    data_processor = ResNet50FeatureExtractor(raw_data_path=raw_data_path,
                                              extracted_features_path=extracted_features_path)
    teaching_set_pool_size = len(data_processor.teaching_images)
    teaching_logic = RandomTeachingLogic(teaching_set_pool_size)
    max_teaching_rounds = 5
    learner_query_text = "Which number do you think this is - 0 or 1?"
    teacher = SimpleTeacher(data_processor=data_processor, teaching_logic=teaching_logic, max_teaching_rounds=max_teaching_rounds, learner_query_text=learner_query_text)
    teacher.perform_teaching_transaction()


simple_teaching_process()