from typing import List, Type

from src.use_cases.use_case import UseCase
from src.metrics import AggregatedL2
from src.metrics import Metric



class ToyUseCase(UseCase):
    """
    Toy use case for testing purposes.

    """
    @property
    def name(self) -> str:
        return "toy_use_case"

    @property
    def target_columns(self) -> List[str]:
        return ['split_win_probs','split_n_bids']

    @property
    def metrics(self) -> List[Type[Metric]]:
        return [AggregatedL2]