import os
from Data_processors.ResNet50FeatureExtractor import ResNet50FeatureExtractor
from Teaching_logics.DataCentricTeachingLogics import RandomTeachingLogic
from Teaching_logics.LearnerCentricTeachingLogics import RandomDisagreeTeachingLogic
from Teachers.BatchTeachers import SimpleBatchTeacher
from EvaluationComponent import BatchTeachingEvaluator
from Virtual_learners.CognitiveLearners import PrototypeLearner
results_directory = "Results"
os.makedirs(results_directory, exist_ok=True)

############ DATA CHOICE #############
DATASET = "Binary_mnist"
human_query_text = "Which number do you think this is - 0 or 1?"
max_teaching_rounds = 5
raw_data_path = os.path.join("Datasets", DATASET)
extracted_features_path = os.path.join("Dataset_features", DATASET)
data_processor = ResNet50FeatureExtractor(raw_data_path=raw_data_path,
                                          extracted_features_path=extracted_features_path)
data_processor_name = "ResNet50"
############ TEACHING LOGIC CHOICE (DATA OR LEARNER CENTRIC) #############
# -----------------------------------use this example of data-centric teaching logic--------------------------------------

# teaching_logic = RandomTeachingLogic(image_candidates_list=data_processor.teaching_images_features,
#                                      image_candidates_labels=data_processor.teaching_images_labels,
#                                      image_candidate_paths=data_processor.teaching_images)
# teaching_logic_name = "random"

# --------------------------------OR use this example of learner-centric teaching logic-----------------------------------------
virtual_learner = PrototypeLearner()
teaching_logic = RandomDisagreeTeachingLogic(image_candidates_list=data_processor.teaching_images_features,
                                     image_candidates_labels=data_processor.teaching_images_labels,
                                     image_candidate_paths=data_processor.teaching_images,
                                     virtual_learner=virtual_learner)
teaching_logic_name = "random_vl_disagree"
############ TEACHING PARADIGM CHOICE (BATCH OR INTERACTIVE) #############
total_teaching_samples = 3
teacher = SimpleBatchTeacher(teaching_logic=teaching_logic, human_query_text=human_query_text,
                             total_teaching_samples=total_teaching_samples)

teaching_paradigm_name = "batch"
############ EVALUATION METHOD CHOICE (BEFORE/AFTER OR CUMULATIVE) #############
eval_results_path = os.path.join(results_directory,f"{DATASET}_{data_processor_name}_{teaching_logic_name}_{teaching_paradigm_name}_results")
evaluator = BatchTeachingEvaluator(teacher=teacher, eval_results_path=eval_results_path, before_test_set_size=3, after_test_set_size = 3)
#
evaluator.evaluate_teaching_framework()
