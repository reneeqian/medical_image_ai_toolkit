from abc import ABC, abstractmethod

class TrainingTaskDefinition(ABC):

    @abstractmethod
    def prepare_training_sample(self, patient_sample):
        """
        Converts PatientSample → model inputs and targets.

        Returns
        -------
        x : torch.Tensor
        y : torch.Tensor
        """
        pass

    @abstractmethod
    def compute_loss(self, prediction, target):
        pass

    def postprocess_prediction(self, prediction):
        return prediction