import abc
from typing import List
import pandas as pd
import numpy as np
from src.models.model import Model

class DummyModel(Model):
    def __init__(self, max_prob : float, random_seed : int = 42):
        self.max_prob = max_prob
        self.random_seed = random_seed
    

    def fit(self, train_input : np.array, train_target : np.array,
                val_input : np.array = None, val_target : np.array = None):
        pass

    def predict(self, test_input: pd.DataFrame) -> np.array:
        np.random.seed(self.random_seed)
        N = test_input.shape[0]
        random_probs = np.random.rand(N)*self.max_prob
        return random_probs

    @property
    def name(self) -> str:
        return 'dummy_model'