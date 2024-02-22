import os
import yaml
import mlflow
from src.logger import logger


class MLFlowClient:
    """
    A class for managing MLflow experiments and logging.

    Attributes:
        _instance (MLFlowClient): The singleton instance of the MLFlowClient.
    """

    _instance = None
    _enabled = None

    def __new__(cls):
        """
        Create a new instance of MLFlowClient if it doesn't exist.

        Returns:
            MLFlowClient: The MLFlowClient instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_experiment_name(self, experiment_name: str):
        """
        Set the active MLFlow experiment by name.

        Args:
            experiment_name (str): The name of the MLFlow experiment.

        Returns:
            None
        """
        logger.info(f"Setting MLFlow experiment name to {experiment_name}")

        if mlflow.get_experiment_by_name(experiment_name) is not None:
            pass
        else:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

    def get_active_run_experiment_name(self) -> str:
        """
        Get the name of the active MLFlow experiment.

        Returns:
            str: The name of the active MLFlow experiment.
        """
        run = mlflow.active_run()
        experiment_id = run.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)
        return experiment.name