from typing import Dict
import numpy as np
import abc

class Metric(abc.ABC):
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def compute(self, targets : np.array, model_preds : np.array) -> float:
        '''
        Computes the metric on the specified targets and predictions.
        
        Args:
            targets (np.array): The true targets.
            model_preds (np.array): The predicted targets.
        
        Returns:
            float: The computed metric.
        
        '''
        pass
