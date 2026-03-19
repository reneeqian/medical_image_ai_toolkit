from abc import ABC, abstractmethod

class TrainingTaskDefinition(ABC):

    @abstractmethod
    def generate_training_samples(self, patient_sample):
        """
        Yields training samples for a patient.

        Each sample is a dict:
            {
                "input": Tensor,
                "target": Tensor or dict
            }
        """
        pass

    @abstractmethod
    def compute_loss(self, prediction, target):
        """
        Task-specific loss computation.
        """
        pass

    def postprocess_prediction(self, prediction):
        return prediction