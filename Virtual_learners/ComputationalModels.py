from AbstractComponents import AbstractVirtualLearner
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class LinearLearner(AbstractVirtualLearner):
    def __init__(self, loss="log_loss"):
        """
        Initialize a linear model with SGD learning
        :param loss: loss functions determine model type
        """
        super().__init__()
        self.model  = SGDClassifier(loss=loss, warm_start=True)


    def fit(self, image, label):
        self.model.partial_fit([image], [label])

    def predict(self, image):
        self.model.predict(image)

    def get_confidence_scores(self, image):
        return self.model.decision_function([image])

    def get_learner_types(self):
        return """
        ‘hinge’ gives a linear SVM.
        ‘log_loss’ gives logistic regression, a probabilistic classifier.
        ‘perceptron’ is the linear loss used by the perceptron algorithm.
        """
