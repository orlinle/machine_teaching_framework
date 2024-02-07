from AbstractComponents import AbstractVirtualLearner
import numpy as np

class PrototypeLearner(AbstractVirtualLearner):
    def __init__(self):
        super().__init__()
        self.prototypes = {}

    def fit(self, image, label):
        if label not in self.prototypes:
            self.prototypes[label] = image
        else:
            self.prototypes[label] = (self.prototypes[label] + image) / 2

    def predict(self, image):
        min_distance = float('inf')
        predicted_label = None
        for label, prototype in self.prototypes.items():
            distance = np.linalg.norm(image - prototype)
            if distance < min_distance:
                min_distance = distance
                predicted_label = label
        return predicted_label

class ExemplarLearner(AbstractVirtualLearner):
    def __init__(self):
        super().__init__()
        self.exemplars = {}

    def fit(self, image, label):
        if label not in self.exemplars:
            self.exemplars[label] = []
        self.exemplars[label].append(image)

    def predict(self, image):
        min_distance = float('inf')
        predicted_label = None
        for label, exemplars in self.exemplars.items():
            for exemplar in exemplars:
                distance = np.linalg.norm(image - exemplar)
                if distance < min_distance:
                    min_distance = distance
                    predicted_label = label
        return predicted_label
