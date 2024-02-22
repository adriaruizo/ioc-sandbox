from typing import Union, Tuple, List, Dict, Type
import pandas as pd
import os
import abc

from src.logger import logger
from src.config import GCS_DATASETS_BUCKET, GCS_DATASETS_PATH
from src.utils import load_parquet_file
from src.models.model import Model
from src.metrics import Metric
from src.mlflow import MLFlowClient
import mlflow

class UseCase(abc.ABC):
    """
    Base class representing a generic use case.

    Attributes:
        DOMAINS (list): A list of supported domain names (adexchange and/or performance).

    """

    DOMAINS = ['adexchange', 'performance']
    def load_benchmark_dataset(self,
                                domain : str,
                                benchmark_id: int,
                                splits  : List[str] = ['train','val','test']) -> Dict[str,pd.DataFrame]:
        """
        Loads benchmark dataset from Google Cloud Storage.

        Args:
            benchmark_id (str): Unique identifier of the benchmark dataset.
            domain (str): The domain of the dataset. It must be one of the supported domains.
            dev_format (bool, optional): Flag indicating whether to load the dataset in development format. Defaults to True.
            splits  (List[str], optional): List of dataset splits to load. Defaults to ['train','val','test'].

        Returns:
            Dict[str,pd.DataFrame]: Returns a single DataFrame if train_val_test_split is False, otherwise returns a tuple containing train, validation, and test DataFrames.

        Raises:
            ValueError: If the specified domain is not in the list of supported domains.

        """
        logger.info(f"Loading benchmark dataset {benchmark_id} from domain {domain}...")
        if domain not in self.DOMAINS:
            raise ValueError(f"Domain {domain} not in {self.DOMAINS}")

        gcs_usecase_path = os.path.join(GCS_DATASETS_BUCKET,
                                        GCS_DATASETS_PATH,
                                        self.name,
                                        domain,
                                        benchmark_id)

        datasets = {}
        for split in splits:
            logger.info(f"Loading {split} split...")
            datasets[split] = load_parquet_file(os.path.join(gcs_usecase_path, f'{split}_split.parquet'))
            self.__check_target_columns(datasets[split])
        return datasets

    def format_input_target_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame]:
        """
        Formats the input and target data from the loaded DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame containing the input and target data.

        Returns:
            Tuple[pd.DataFrame]: A tuple containing the input and target DataFrames.

        """
        return data.drop(columns=self.target_columns).to_numpy(), data[self.target_columns].to_numpy()


    def evaluate(self, domain: str, benchmark_id: int, model_class : Type[Model], log_mlflow : bool = False, **model_params):
        """
        Evaluates the model on the specified domain and benchmark_id.

        Args:
            domain (str): The domain of the dataset to evaluate the model on.
            benchmark_id (int): The benchmark_id of the dataset to evaluate the model on.
            model (Model): The model to evaluate.
            log_mlflow (bool, optional): Flag indicating whether to log the evaluation results to MLFlow. Defaults to False.
            model_params (Dict): The parameters to pass to the model.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results.

        """
        logger.info(f"Evaluating model on domain {domain} and benchmark_id {benchmark_id}...")
        logger.info(f"Logging to MLFlow: {log_mlflow}")
        logger.info('Initializing model...')
        model = model_class(**model_params)
        if(log_mlflow):
            MLFlowClient().set_experiment_name(self.name)
            with mlflow.start_run():
                mlflow.log_params(model_params)
                mlflow.log_params({'use_case': self.name})
                mlflow.log_params({'domain': domain, 'benchmark_id': benchmark_id})
                mlflow.log_params({'model': model.name})
                print(model.name)
                metrics = self.__evaluate(domain, benchmark_id, model, **model_params)
                mlflow.log_metrics(metrics)
                return metrics
        else:
            return self.__evaluate(domain, benchmark_id, model, **model_params)

    def __evaluate(self, domain: str, benchmark_id: int, model : Model, **model_params):
        """
        Evaluates the model on the specified domain and benchmark_id.

        Args:
            domain (str): The domain of the dataset to evaluate the model on.
            benchmark_id (int): The benchmark_id of the dataset to evaluate the model on.
            model (Model): The model to evaluate.
            model_params (Dict): The parameters to pass to the model.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results.

        """
        datasets = self.load_benchmark_dataset(domain, benchmark_id)


        # Fit the model on the training data and validate potential hyperparameters on the validation data
        train_input, train_target = self.format_input_target_data(datasets['train'])
        val_input, val_target = self.format_input_target_data(datasets['val'])

        logger.info('Fitting model with train and val data')
        model.fit(train_input, train_target, val_input, val_target)

        # Make predictions on the test data
        logger.info('Making predictions on test data')
        test_input, test_target = self.format_input_target_data(datasets['test'])
        model_predictions = model.predict(test_input)

        # Evaluate the predictions according to the use case's metrics
        logger.info('Evaluating predictions with use-case metrics...')
        results = {}
        for metric_cls in self.metrics:
            metric = metric_cls()
            results[metric.name] = metric.compute(test_target, model_predictions)
        return results

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Abstract property representing the name of the use case.

        Returns:
            str: The name of the use case.
        """
        pass

    @property
    @abc.abstractmethod
    def metrics(self) -> List[Type[Metric]]:
        """
        Abstract property representing the metrics used to evaluate the model.

        Returns:
            List[Type[Metric]]: A list of metric classes.
        """
        pass

    @property
    @abc.abstractmethod
    def target_columns(self) -> List[str]:
        """
        Abstract property representing the name of the column containing the model predictions.

        Returns:
            List[str]: A list of column names containing the model predictions.
        """
        pass

    def __check_target_columns(self, targets: pd.DataFrame) -> None:
        """
        Checks if the target columns are present in the DataFrame.

        Args:
            targets (pd.DataFrame): The DataFrame containing the target columns.

        Raises:
            ValueError: If any of the target columns are not present in the DataFrame.

        """
        for col in self.target_columns:
            if col not in targets.columns:
                raise ValueError(f"Column {col} not found in the loaded DataFrame")