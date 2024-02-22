import abc
from typing import Dict, Any, List
import numpy as np

class Model(abc.ABC):
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        '''
        Abstract property representing the name of the model.

        Returns:
            str: The name of the model.

        '''
        pass

    @abc.abstractmethod
    def fit(self, train_input : np.array, train_target : np.array, 
                    val_input : np.array = None, val_target : np.array = None):
        '''
        Fits the model on the training data and validates potential hyperparameters on the validation data.

        Args:
            train_dataset (pd.DataFrame): The training dataset.
            val_dataset (pd.DataFrame, optional): The validation dataset. Defaults to None.

        '''
        pass

    @abc.abstractmethod
    def predict(self, test_input : np.array) -> np.array:
        '''
        Makes predictions on the test data.

        Args:
            test_dataset (pd.DataFrame): The test dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the predictions.

        ''' 
        pass