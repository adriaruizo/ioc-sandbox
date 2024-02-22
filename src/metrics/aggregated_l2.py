from typing import Dict
import numpy as np
from src.metrics.metric import Metric

class AggregatedL2(Metric):
    @property
    def name(self) -> str:
        return "aggregated_l2"

    def compute(self, targets : np.array, model_preds : np.array) -> float:
        '''
        Computes the aggregated L2 metric on the specified targets and predictions.

        Args:
            targets (np.array): The true targets.
            model_preds (np.array): The predicted targets.
        
        Returns:
            float: The computed aggregated L2 metric.
        
        '''
        target_vals = targets[:, 0]
        samples = targets[:, 1]

        return np.sqrt(np.sum( ((target_vals - model_preds)**2)*samples) / np.sum(samples))
